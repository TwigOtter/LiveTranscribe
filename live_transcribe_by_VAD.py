"""
Live transcription using Voice Activity Detection (VAD) to trigger transcription.

Instead of transcribing on fixed time intervals with overlapping windows,
this approach accumulates audio continuously and only transcribes when:
1. Silence is detected for 1.5+ seconds, OR
2. A maximum duration (20s) is reached without silence

This produces clean, sentence-based transcriptions with no duplicates.
"""

import argparse
import json
import queue
import re
import sys
import threading
from datetime import datetime

import requests
import sounddevice as sd

from audio_pipeline import AudioPipeline, PendingAudio
from transcriber import Transcriber


def _load_settings() -> dict:
    try:
        with open("settings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print("[warning] settings.json not found. Using built-in defaults.")
        return {}
    except json.JSONDecodeError as e:
        print(f"[warning] settings.json is invalid JSON: {e}. Using built-in defaults.")
        return {}


def list_devices_and_exit():
    print("Available input devices:")
    for idx, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:
            print(f"{idx:>3}: {info['name']}  | hostapi={info['hostapi']}  | sr={info.get('default_samplerate')}")
    sys.exit(0)


def find_input_device(device_name_substring: str | None) -> int:
    if not device_name_substring:
        return sd.default.device[0]
    name_lower = device_name_substring.lower()
    for idx, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0 and name_lower in info["name"].lower():
            return idx
    raise RuntimeError(
        f"Could not find input device containing '{device_name_substring}'. "
        f"Run with --list-devices to see options."
    )


def apply_word_replacements(text: str, replacements: dict) -> str:
    for search_term, replacement in replacements.items():
        pattern = r'\b' + re.escape(search_term) + r'\b'
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def post_to_streamerbot(text: str, speaker: str, url: str, action_id: str = "", action_name: str = "", extra_args: dict = None):
    data = {
        "action": {"id": action_id, "name": action_name},
        "args": {"speaker": speaker, "message": text, **(extra_args or {})},
    }
    try:
        response = requests.post(url, headers={"Content-Type": "application/json"}, json=data, timeout=5)
        print(f"POST RESPONSE: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[warning] Failed to post to StreamerBot: {e}")


def log_to_jsonl(text: str, file_path: str, speaker: str):
    entry = {"ts": datetime.now().isoformat() + "Z", "speaker": speaker, "text": text.strip()}
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _transcription_worker(
    transcriber: Transcriber,
    transcription_queue: "queue.Queue[PendingAudio | None]",
    word_replacements: dict,
    speaker_name: str,
    url: str,
    action_id: str,
    action_name: str,
    output_file: str,
    debug: bool,
    sample_rate: int,
):
    count = 0
    while True:
        pending = transcription_queue.get()
        if pending is None:
            break
        count += 1
        print(f"\n[transcribe] *** TRANSCRIPTION EVENT #{count} ***", flush=True)
        print(f"[transcribe] Processing {len(pending.audio)/sample_rate:.1f}s of audio…", flush=True)
        text = transcriber.transcribe(pending.audio)
        if text:
            text = apply_word_replacements(text, word_replacements)
            print(f"[result] {text}")
            extra = {"berries": "true"} if pending.is_berries else None
            post_to_streamerbot(text, speaker_name, url, action_id, action_name, extra_args=extra)
            log_to_jsonl(text, output_file, speaker_name)
        elif debug:
            print("[result] No speech detected in accumulated audio.")


def main():
    settings = _load_settings()
    word_replacements = settings.get("replacements", {})

    ap = argparse.ArgumentParser(description="Live transcription using faster-whisper with VAD-triggered batching.")
    ap.add_argument("-n", "--speaker-name",        default=settings.get("speaker_name", "TwigOtter"),          help="Name of the speaker. Overrides settings.json.")
    ap.add_argument("--url",                        default=settings.get("streamerbot_url", "http://127.0.0.1:7474/DoAction"), help="StreamerBot webhook URL. Overrides settings.json.")
    ap.add_argument("--action-id",                  default=settings.get("streamerbot_action_id", "05d6af77-9ed2-4771-be77-1e666955873d"), help="StreamerBot action ID to invoke. Overrides settings.json.")
    ap.add_argument("--action-name",                default=settings.get("streamerbot_action_name", "LiveTranscribe"), help="StreamerBot action name to invoke. Overrides settings.json.")
    ap.add_argument("-m", "--model",                default=settings.get("model", "large-v3"),                  help="faster-whisper model size or path. Overrides settings.json.")
    ap.add_argument("-d", "--device-name",          default=settings.get("device_name", None),                  help="Substring of input device name. Use --list-devices to discover. Overrides settings.json.")
    ap.add_argument("-l", "--list-devices",         action="store_true",                                         help="List input devices and exit.")
    ap.add_argument("-r", "--sample-rate",          type=int,   default=settings.get("sample_rate", 16000),     help="Target sample rate for ASR (Hz). Overrides settings.json.")
    ap.add_argument("--language",                   default=settings.get("language", "en"),                     help="Force language code (e.g., en). Overrides settings.json.")
    ap.add_argument("-c", "--compute-type",         default=settings.get("compute_type", "float16"),            help="e.g., int8, float16, float32. Overrides settings.json.")
    ap.add_argument("-s", "--silence-duration-ms",  type=float, default=settings.get("silence_duration_ms", 1500.0), help="Duration of silence (ms) to trigger transcription. Overrides settings.json.")
    ap.add_argument("-b", "--max-buffer-seconds",   type=float, default=settings.get("max_buffer_seconds", 15.0),   help="Maximum seconds to accumulate before forcing transcription. Overrides settings.json.")
    ap.add_argument("-v", "--vad-threshold",        type=float, default=settings.get("vad_threshold", 0.5),     help="VAD probability threshold (0-1). Overrides settings.json.")
    ap.add_argument("-x", "--debug",                action="store_true", default=settings.get("debug", False),  help="Enable verbose debug logging. Overrides settings.json.")
    ap.add_argument("-o", "--output-file",          default=settings.get("output_file", "live_transcript.jsonl"), help="Output JSONL file for transcriptions. Overrides settings.json.")
    ap.add_argument("--berries-key",                default=settings.get("berries_key", "f13"),                  help="Hotkey to hold for Berries recording (e.g. 'f9', 'scroll_lock'). Overrides settings.json.")
    args = ap.parse_args()

    if args.list_devices:
        list_devices_and_exit()

    try:
        device_idx = find_input_device(args.device_name)
    except RuntimeError as e:
        print(str(e))
        list_devices_and_exit()

    print(f"[info] Using input device index {device_idx}: {sd.query_devices(device_idx)['name']}")
    print(f"[info] Silence threshold: {args.silence_duration_ms}ms | Max buffer: {args.max_buffer_seconds}s | VAD threshold: {args.vad_threshold}")

    transcriber = Transcriber(
        model_name=args.model,
        compute_type=args.compute_type,
        language=args.language,
        vad_threshold=args.vad_threshold,
    )

    pipeline = AudioPipeline(
        device_idx=device_idx,
        sample_rate=args.sample_rate,
        vad_threshold=args.vad_threshold,
        silence_duration_ms=args.silence_duration_ms,
        max_buffer_seconds=args.max_buffer_seconds,
        berries_key=args.berries_key,
        debug=args.debug,
    )

    pipeline.start()

    transcription_queue: queue.Queue[PendingAudio | None] = queue.Queue()
    worker = threading.Thread(
        target=_transcription_worker,
        args=(
            transcriber,
            transcription_queue,
            word_replacements,
            args.speaker_name,
            args.url,
            args.action_id,
            args.action_name,
            args.output_file,
            args.debug,
            args.sample_rate,
        ),
        daemon=True,
        name="transcription-worker",
    )
    worker.start()
    print("[info] Listening… Press Ctrl+C to stop.")

    try:
        while True:
            pending = pipeline.get_transcription_audio(timeout=1.0)
            if pending is None:
                continue
            transcription_queue.put(pending)
    except KeyboardInterrupt:
        print("\n[info] Stopping…")
    finally:
        transcription_queue.put(None)
        worker.join(timeout=30)
        pipeline.stop()


if __name__ == "__main__":
    main()
