import queue
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
from pynput import keyboard as kb
from silero_vad import load_silero_vad, get_speech_timestamps


@dataclass
class PendingAudio:
    audio: np.ndarray
    is_berries: bool


class AudioPipeline:
    def __init__(
        self,
        device_idx: int,
        sample_rate: int,
        vad_threshold: float,
        silence_duration_ms: float,
        max_buffer_seconds: float,
        berries_key: str,
        debug: bool = False,
    ):
        self._device_idx = device_idx
        self._sample_rate = sample_rate
        self._vad_threshold = vad_threshold
        self._silence_duration_ms = silence_duration_ms
        self._max_buffer_samples = int(sample_rate * max_buffer_seconds)
        self._vad_check_interval_samples = int(sample_rate * 0.1)
        self._min_audio_samples = sample_rate // 4
        self._debug = debug

        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._pending_queue: queue.Queue[PendingAudio] = queue.Queue()

        self._accumulated_audio = np.array([], dtype=np.float32)
        self._accumulated_audio_lock = threading.Lock()
        self._speech_detected_in_buffer = False

        self._stop_event = threading.Event()
        self._hotkey_held = threading.Event()

        print("[info] Loading Silero VAD model…")
        self._vad_model = load_silero_vad()

        self._berries_target_key = self._parse_key(berries_key)
        self._berries_key_label = berries_key.upper()

        self._stream: sd.InputStream | None = None
        self._ingest_thread: threading.Thread | None = None
        self._hotkey_listener: kb.Listener | None = None

    @staticmethod
    def _parse_key(key_str: str):
        key_str_norm = key_str.strip().lower().replace('-', '_').replace(' ', '_')
        if hasattr(kb.Key, key_str_norm):
            return getattr(kb.Key, key_str_norm)
        if len(key_str) == 1:
            return kb.KeyCode.from_char(key_str)
        raise ValueError(
            f"Unknown berries_key: {key_str!r}. "
            f"Use a special key name (e.g. 'f9', 'scroll_lock') or a single character."
        )

    def start(self):
        self._ingest_thread = threading.Thread(target=self._ingest_loop, daemon=True)
        self._ingest_thread.start()

        self._hotkey_listener = kb.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._hotkey_listener.start()
        print(f"[info] Berries hotkey: hold {self._berries_key_label} to record, release to send to Berries.")

        self._stream = sd.InputStream(
            device=self._device_idx,
            channels=1,
            samplerate=self._sample_rate,
            blocksize=1024,
            dtype="float32",
            callback=self._audio_callback,
            latency="low",
        )
        self._stream.start()

    def stop(self):
        self._stop_event.set()
        if self._hotkey_listener:
            try:
                self._hotkey_listener.stop()
            except Exception:
                pass
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass

    def get_transcription_audio(self, timeout: float = 1.0) -> PendingAudio | None:
        try:
            return self._pending_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # --- Internal ---

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        mono = indata[:, 0] if indata.shape[1] == 1 else np.mean(indata, axis=1)
        try:
            self._audio_queue.put_nowait(mono.astype(np.float32))
        except queue.Full:
            pass

    def _on_key_press(self, key):
        if key == self._berries_target_key and not self._hotkey_held.is_set():
            with self._accumulated_audio_lock:
                if self._speech_detected_in_buffer and len(self._accumulated_audio) > self._min_audio_samples:
                    self._pending_queue.put(PendingAudio(audio=self._accumulated_audio.copy(), is_berries=False))
                self._accumulated_audio = np.array([], dtype=np.float32)
                self._speech_detected_in_buffer = False
            self._hotkey_held.set()
            print(f"[hotkey] {self._berries_key_label} held — recording for Berries…", flush=True)

    def _on_key_release(self, key):
        if key == self._berries_target_key and self._hotkey_held.is_set():
            self._hotkey_held.clear()
            with self._accumulated_audio_lock:
                audio_snapshot = self._accumulated_audio.copy()
                self._accumulated_audio = np.array([], dtype=np.float32)
                self._speech_detected_in_buffer = False
            if len(audio_snapshot) > self._min_audio_samples:
                self._pending_queue.put(PendingAudio(audio=audio_snapshot, is_berries=True))
                print(f"[hotkey] Released — sending {len(audio_snapshot)/self._sample_rate:.1f}s to Berries.", flush=True)
            else:
                print("[hotkey] Released — not enough audio to send.", flush=True)

    def _ingest_loop(self):
        last_vad_check_time = time.time()
        silence_start_time = None
        check_count = 0
        debug = self._debug

        print("[ingest] Thread started", flush=True)

        while not self._stop_event.is_set():
            # Accumulate incoming audio
            try:
                audio_block = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                audio_block = None

            if audio_block is not None:
                with self._accumulated_audio_lock:
                    self._accumulated_audio = np.concatenate([self._accumulated_audio, audio_block])
                    new_len = len(self._accumulated_audio)
                    if debug:
                        print(f"[ingest] Buffer: {new_len} samples ({new_len/self._sample_rate:.2f}s)", flush=True)

                    # Hard cap — skip during hotkey hold since user controls duration
                    if not self._hotkey_held.is_set() and new_len > self._max_buffer_samples:
                        print("[vad] Max buffer duration reached. Forcing transcription.", flush=True)
                        audio_snapshot = self._accumulated_audio.copy()
                        self._accumulated_audio = np.array([], dtype=np.float32)
                        self._speech_detected_in_buffer = False
                        silence_start_time = None
                        self._pending_queue.put(PendingAudio(audio=audio_snapshot, is_berries=False))

            # Periodic VAD silence check — skip during hotkey hold
            now = time.time()
            if self._hotkey_held.is_set() or now - last_vad_check_time < 0.1:
                continue

            last_vad_check_time = now
            check_count += 1

            with self._accumulated_audio_lock:
                buffer_len = len(self._accumulated_audio)
                if buffer_len <= self._vad_check_interval_samples:
                    if debug:
                        print(f"[vad-check #{check_count}] Not enough audio yet: {buffer_len} samples", flush=True)
                    continue
                recent_chunk = self._accumulated_audio[-self._vad_check_interval_samples:].copy()

            # VAD check runs outside the lock to minimise contention
            try:
                vad_audio = recent_chunk / max(abs(recent_chunk).max(), 1e-9)

                speech_timestamps = get_speech_timestamps(
                    vad_audio,
                    self._vad_model,
                    sampling_rate=self._sample_rate,
                    threshold=self._vad_threshold,
                    min_speech_duration_ms=50,
                    min_silence_duration_ms=50,
                    return_seconds=False,
                )
                has_speech = len(speech_timestamps) > 0

                if debug:
                    print(f"[vad-check #{check_count}] has_speech={has_speech}", flush=True)

                if has_speech:
                    with self._accumulated_audio_lock:
                        self._speech_detected_in_buffer = True
                    silence_start_time = None
                else:
                    if silence_start_time is None:
                        silence_start_time = now
                    silence_duration_ms = (now - silence_start_time) * 1000

                    if debug:
                        print(f"[vad-check #{check_count}] Silence: {silence_duration_ms:.0f}ms / {self._silence_duration_ms:.0f}ms", flush=True)

                    if silence_duration_ms >= self._silence_duration_ms:
                        with self._accumulated_audio_lock:
                            audio_len = len(self._accumulated_audio)
                            if audio_len > self._min_audio_samples:
                                if self._speech_detected_in_buffer:
                                    audio_snapshot = self._accumulated_audio.copy()
                                    self._accumulated_audio = np.array([], dtype=np.float32)
                                    self._speech_detected_in_buffer = False
                                    silence_start_time = None
                                    print(f"[vad] Silence detected ({silence_duration_ms:.0f}ms). Triggering transcription.", flush=True)
                                    self._pending_queue.put(PendingAudio(audio=audio_snapshot, is_berries=False))
                                else:
                                    if debug:
                                        print(f"[vad-check #{check_count}] No speech in buffer, flushing silence.", flush=True)
                                    self._accumulated_audio = np.array([], dtype=np.float32)
                                    silence_start_time = None

            except Exception as e:
                import traceback
                print(f"[vad] Error during VAD check: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
