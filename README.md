# Pedal Dictation

Free, local push-to-talk voice dictation for Windows. Hold a USB foot pedal (or hotkey), speak, release — text appears wherever your cursor is.

Uses OpenAI Whisper (via faster-whisper) running locally on your NVIDIA GPU, with optional Groq LLM post-processing for punctuation cleanup.

## Why

Commercial dictation tools (Whisper Flow and similar) charge a monthly subscription and route audio through their servers. This project is a minimal, privacy-respecting alternative: all transcription runs locally on your GPU, the only optional network call is to Groq for punctuation cleanup, and the entire app is a single Python script you can read in one sitting.

## Features

- **Push-to-talk** — hold to record, release to transcribe and paste
- **Local AI** — Whisper medium model runs on your GPU via CUDA (no cloud for transcription)
- **Punctuation cleanup** — optional Groq API pass fixes formatting (free tier, 14,400 req/day)
- **Keyboard layout independent** — works on any layout (English, Cyrillic, etc.)
- **Clipboard preserved** — your clipboard is restored after paste
- **Voice commands** — say "send it", "press enter", or "hit enter" to auto-submit
- **Paste last** — Alt+Shift+Z pastes the last dictation again
- **Runs hidden** — no terminal window, auto-starts with Windows

## Requirements

- Windows 10/11
- NVIDIA GPU (CUDA-capable)
- Python 3.13+
- USB foot pedal configured to send `Ctrl+Shift+F5` (or use the hotkey on your keyboard)

## Installation

```bash
pip install --target vendor -r requirements.txt
```

This installs all dependencies (including the NVIDIA CUDA runtime libraries) into a `vendor/` folder next to the script. The script prepends `vendor/` to `sys.path` at startup, so it doesn't matter which Python on the machine ends up launching it — the deps it finds are always the ones in `vendor/`. This also avoids per-user `site-packages` issues with sandboxed/AppContainer launch contexts.

The Whisper model (~1.5GB) downloads on first run into `~/.cache/huggingface/hub/` and is reused on subsequent launches.

## Usage

### Run with terminal (for debugging):
```bash
python pedal_dictation.py
```

### Run hidden (no window):
```bash
wscript.exe pedal_dictation.vbs
```

### Auto-start with Windows:
Place a shortcut to `pedal_dictation.vbs` in your Startup folder:
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\`

## Hotkeys

| Action | Shortcut |
|--------|----------|
| Record (push-to-talk) | `Ctrl+Shift+F5` (pedal) |
| Paste last dictation | `Alt+Shift+Z` |

## Voice Commands

Say these at the end of your dictation:
- **"send it"** — pastes text + presses Enter
- **"press enter"** — same as above
- **"hit enter"** — same as above

If said alone (without other text), just presses Enter.

## Configuration

### Custom Dictionary / Snippets

Create a `dictionary.json` next to the script to fix consistently mis-heard words or expand snippets:

```json
{
  "anthropc": "Anthropic",
  "claud": "Claude",
  "my email": "you@example.com"
}
```

- Case-insensitive whole-word matching
- Applied after Whisper, before Groq cleanup
- Missing file is fine — the feature is a no-op without it
- See `dictionary.example.json` for a template
- **Note:** Avoid adding entries for voice command phrases ("send it", "press enter", "hit enter") as substitutions run before command detection

### Model and language

Copy `config.example.json` to `config.json` and edit as needed:

```json
{
  "model_size": "medium",
  "language": "en",
  "groq_model": "llama-3.1-8b-instant",
  "record_hotkey": [162, 160, 116],
  "paste_last_hotkey": [164, 160, 90]
}
```

All keys are optional — missing keys fall back to defaults. Missing file is fine too.

Hotkeys are arrays of Windows virtual key codes. The non-modifier key (anything outside 160-165) is used as the trigger for push-to-talk release detection. Defaults: `[162, 160, 116]` = Ctrl+Shift+F5, `[164, 160, 90]` = Alt+Shift+Z.

### Groq API key (optional)

Punctuation cleanup needs a free Groq API key from https://console.groq.com. Without it the app still works — you just get raw Whisper output.

Provide the key in one of two ways:

**Option A — system environment variable (recommended for autostart):**
```powershell
setx GROQ_API_KEY "gsk_..."
```
Then sign out and back in (or reboot) so `pythonw.exe` picks it up.

**Option B — `.env` file next to the script:**
Copy `.env.example` to `.env` and fill in the key. The script reads it on startup. `.env` is gitignored.

## USB Pedal Setup

Any USB pedal that can be programmed to send a key combination works. Recommended: PCsensor foot switches with the FootSwitch software. Configure the pedal to send `Ctrl+Shift+F5`.

## Architecture

Single Python script. No frameworks, no config files, no database.

1. **pynput** — global hotkey listener (layout-independent via virtual key codes)
2. **sounddevice** — microphone capture
3. **faster-whisper** — local Whisper inference on CUDA
4. **Groq API** — optional LLM punctuation cleanup
5. **pyperclip + pynput** — clipboard paste into active window

## License

[MIT](LICENSE)
