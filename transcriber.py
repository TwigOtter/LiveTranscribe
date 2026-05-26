import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    def __init__(self, model_name: str, compute_type: str, language: str, vad_threshold: float, device: str = "auto"):
        print(f"[info] Loading faster-whisper model: {model_name} (device={device}, compute_type={compute_type}) …")
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self._language = language
        self._vad_threshold = vad_threshold

    def transcribe(self, audio: np.ndarray) -> str:
        segments_gen, _ = self._model.transcribe(
            audio,
            language=self._language,
            vad_filter=True,
            vad_parameters=dict(
                threshold=self._vad_threshold,
                min_silence_duration_ms=2000,
            ),
            beam_size=5,
            best_of=5,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        return " ".join(seg.text.strip() for seg in segments_gen).strip()
