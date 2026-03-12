# LiveTranscribe

Real-time audio transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [Silero VAD](https://github.com/snakers4/silero-vad). Audio is captured continuously and transcription is triggered automatically when silence is detected — producing clean, sentence-based output with no duplicates.

Optionally posts each transcription to a [StreamerBot](https://streamer.bot/) webhook.

---

## Requirements

- Python 3.10+
- A CUDA-capable GPU is strongly recommended for the `large-v3` model. CPU inference works but is significantly slower — use `tiny` or `base` if running on CPU.

---

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** `torch` is a large dependency (~2 GB). If you already have a CUDA-compatible PyTorch installed, you can skip it in requirements.txt.

The Whisper model weights are downloaded automatically on first run. `large-v3` is ~3 GB.

---

## Configuration

All user-facing settings live in `settings.json`. Open it in Notepad (or any text editor) and edit the values — no coding required. Restart the script after saving.

```json
{
  "speaker_name": "YourNameHere",
  "streamerbot_url": "http://127.0.0.1:7474/DoAction",
  "replacements": {
    "barry's": "Berries'",
    "barry": "Berries",
    "your word here": "Corrected Word"
  }
}
```

| Key | Default | Description |
|-----|---------|-------------|
| `speaker_name` | `"TwigOtter"` | Your name, sent to StreamerBot with each transcription |
| `streamerbot_url` | `"http://..."` | The HTTP URL of your StreamerBot webhook |
| `streamerbot_action_name` | `"LiveTranscribe"` | The name of the StreamerBot action to invoke |
| `replacements` | `{}` | Words Whisper mishears → what they should say. Matching is case-insensitive; replacement is applied exactly as written. |
| `model` | `"large-v3"` | Whisper model size. Use `"base"` or `"small"` if running on CPU. |
| `device_name` | `null` | Substring of your audio input device name. `null` uses the system default. Run `--list-devices` to find yours. |
| `language` | `"en"` | Language code to force (e.g. `"en"`, `"ja"`). |
| `compute_type` | `"float16"` | Model precision. Use `"int8"` for CPU. |
| `sample_rate` | `16000` | Audio sample rate in Hz. |
| `silence_duration_ms` | `1500` | Milliseconds of silence before transcription is triggered. |
| `max_buffer_seconds` | `15` | Max seconds of audio to buffer before forcing transcription. |
| `vad_threshold` | `0.5` | Voice detection sensitivity (0–1). Higher = stricter. |
| `output_file` | `"live_transcript.jsonl"` | File to write transcription logs to. |
| `debug` | `false` | Set to `true` for verbose logging. |

If `settings.json` is missing or broken, the script will warn you and fall back to built-in defaults.

### StreamerBot Integration

If the StreamerBot endpoint is unreachable the script will print a warning and keep transcribing — it will not crash.

---

## Usage

```
python live_transcribe_by_VAD.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `-n, --speaker-name` | from `settings.json` | Speaker name sent to StreamerBot |
| `--url` | from `settings.json` | StreamerBot webhook URL |
| `-m, --model` | `large-v3` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `-d, --device-name` | system default | Substring of the audio input device name |
| `-l, --list-devices` | — | Print available input devices and exit |
| `-r, --sample-rate` | `16000` | Audio sample rate in Hz |
| `--language` | `en` | Force language code (e.g. `en`, `ja`). Auto-detects if omitted |
| `-c, --compute-type` | `float16` | Model precision (`int8`, `int8_float16`, `float16`, `float32`) |
| `-s, --silence-duration-ms` | `1500` | Milliseconds of silence required to trigger transcription |
| `-b, --max-buffer-seconds` | `15` | Max seconds to accumulate before forcing transcription |
| `-v, --vad-threshold` | `0.5` | VAD sensitivity (0–1, higher = stricter) |
| `-o, --output-file` | `live_transcript.jsonl` | Output file path |
| `-x, --debug` | — | Enable verbose debug logging |

### Common Examples

List available audio input devices:
```bash
python live_transcribe_by_VAD.py --list-devices
```

Transcribe from a specific device with a speaker name:
```bash
python live_transcribe_by_VAD.py --device-name "CABLE Output" --speaker-name MyName
```

Use a smaller model for CPU:
```bash
python live_transcribe_by_VAD.py --model base --compute-type int8
```

Increase silence threshold to wait longer before transcribing:
```bash
python live_transcribe_by_VAD.py --silence-duration-ms 2500
```

---

## License

MIT License

Copyright (c) 2025 TwigOtter

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
