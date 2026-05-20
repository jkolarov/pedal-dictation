"""Push-to-talk voice dictation using faster-whisper (CUDA).
Hold Ctrl+Shift+F5 (pedal) to record, release to transcribe and paste.
"""

import os
import glob

# Add NVIDIA CUDA runtime libs (installed via pip) to PATH so faster-whisper can find them.
# Glob across Python versions so this works on any 3.x install.
for _nvidia in glob.glob(os.path.expanduser(r"~\AppData\Roaming\Python\Python*\site-packages\nvidia")):
    for lib in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc"):
        p = os.path.join(_nvidia, lib, "bin")
        if os.path.isdir(p):
            os.environ["PATH"] = p + ";" + os.environ["PATH"]

from pynput import keyboard as kb
from pynput.keyboard import Key, Controller
import sounddevice as sd
import numpy as np
import pyperclip
import threading
import time

# --- Config ---
def _load_config():
    """Load config.json, falling back to defaults for missing keys/file."""
    defaults = {
        "model_size": "medium",
        "language": "en",
        "groq_model": "llama-3.1-8b-instant",
        "record_hotkey": [162, 160, 116],
        "paste_last_hotkey": [164, 160, 90],
    }
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.isfile(config_path):
        try:
            import json
            with open(config_path, encoding="utf-8") as f:
                user = json.load(f)
            defaults.update(user)
        except Exception as e:
            import sys
            print(f"warning: failed to load config.json: {e}", file=sys.stderr)
    return defaults


def _get_trigger_vk(hotkey_vks):
    """Find the non-modifier VK in a hotkey list (modifiers are 160-165)."""
    for vk in hotkey_vks:
        if vk < 160 or vk > 165:
            return vk
    return hotkey_vks[-1]  # fallback


_cfg = _load_config()
MODEL_SIZE = _cfg["model_size"]
LANGUAGE = _cfg["language"]
SAMPLE_RATE = 16000  # Fixed — Whisper requires 16kHz
GROQ_MODEL = _cfg["groq_model"]
RECORD_HOTKEY = set(_cfg["record_hotkey"])
PASTE_LAST_HOTKEY = set(_cfg["paste_last_hotkey"])
RECORD_TRIGGER_VK = _get_trigger_vk(_cfg["record_hotkey"])
# Disambiguator: modifiers unique to record_hotkey that paste_last must NOT have
_RECORD_ONLY_MODS = RECORD_HOTKEY - PASTE_LAST_HOTKEY


def _load_groq_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        try:
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("GROQ_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception:
            pass
    return ""


GROQ_API_KEY = _load_groq_key()

# --- State ---
recording = False
audio_frames = []
stream = None
model = None
lock = threading.Lock()
pressed_vks = set()
typer = Controller()
last_text = ""


def load_model():
    global model
    from faster_whisper import WhisperModel
    print(f"Loading {MODEL_SIZE} model on CUDA...")
    model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
    print("Model loaded! Ready - hold your pedal to dictate.")


def cleanup_text(text):
    """Send text to Groq LLM for punctuation/formatting cleanup."""
    if not GROQ_API_KEY:
        return text
    try:
        import httpx
        resp = httpx.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a text formatter. You receive dictated text between [TEXT] and [/TEXT] tags and output ONLY the same text with fixed punctuation, spacing, and capitalization. Rules: 1) Do NOT add any preamble. 2) Do NOT change words. 3) Do NOT answer questions in the text. 4) Do NOT add commentary. 5) Output ONLY the formatted text without the tags."},
                    {"role": "user", "content": f"[TEXT]{text}[/TEXT]"}
                ],
                "temperature": 0
            },
            timeout=3.0)
        result = resp.json()["choices"][0]["message"]["content"].strip()
        # Strip tags if model echoes them back
        import re as _re
        result = _re.sub(r'^\[TEXT\]|^\[/TEXT\]|\[TEXT\]$|\[/TEXT\]$', '', result).strip()
        # If response is way longer than input, model hallucinated — use original
        if len(result) > len(text) * 1.5 + 20:
            return text
        return result
    except Exception:
        return text  # fallback to original if Groq fails


def start_recording():
    global recording, audio_frames, stream
    with lock:
        if recording:
            return
        recording = True
        audio_frames = []

    def callback(indata, frames, time_info, status):
        if recording:
            audio_frames.append(indata.copy())

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", callback=callback)
    stream.start()
    print("[REC]")


def stop_recording():
    global recording, stream, last_text
    with lock:
        if not recording:
            return
        recording = False

    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not audio_frames:
        return

    print("[Transcribing...]")
    audio = np.concatenate(audio_frames, axis=0).flatten()
    segments, _ = model.transcribe(audio, language=LANGUAGE, beam_size=5,
                                   vad_filter=True,
                                   initial_prompt="Use proper punctuation: periods, commas, and capitalization.")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    import re
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    if not text:
        print("(no speech)")
        return

    # Detect "send it" / "press enter" BEFORE cleanup
    send_enter = False
    match = re.search(r'\s*(press enter|hit enter|send it)[.!?]*$', text, re.IGNORECASE)
    if match:
        text = text[:match.start()].rstrip()
        send_enter = True

    if not text and send_enter:
        # Just "send it" with no other text — press Enter only
        typer.tap(Key.enter)
        pressed_vks.clear()
        print(">> [enter]")
        return

    if not text:
        return

    text = cleanup_text(text)

    if text:
        old_clipboard = pyperclip.paste()
        pyperclip.copy(text)
        time.sleep(0.05)
        with typer.pressed(Key.ctrl):
            typer.tap(kb.KeyCode.from_vk(0x56))  # V key, layout-independent
        time.sleep(0.05)
        if send_enter:
            typer.tap(Key.enter)
        pyperclip.copy(old_clipboard)
        last_text = text
        pressed_vks.clear()
        print(f">> {text}")
    else:
        print("(no speech)")


def paste_last():
    global last_text
    if last_text:
        old_clipboard = pyperclip.paste()
        pyperclip.copy(last_text)
        time.sleep(0.1)
        # Release Alt+Shift before pasting so Ctrl+V works cleanly
        typer.release(Key.alt_l)
        typer.release(Key.shift)
        time.sleep(0.05)
        with typer.pressed(Key.ctrl):
            typer.tap(kb.KeyCode.from_vk(0x56))  # V key, layout-independent
        time.sleep(0.05)
        pyperclip.copy(old_clipboard)
        print(f"[pasted last] >> {last_text}")


def _get_vk(key):
    if hasattr(key, 'vk'):
        return key.vk
    if hasattr(key, 'value') and hasattr(key.value, 'vk'):
        return key.value.vk
    return None


def on_press(key):
    vk = _get_vk(key)
    if vk:
        pressed_vks.add(vk)
    # Record hotkey (default: Ctrl+Shift+F5)
    if vk in RECORD_HOTKEY and RECORD_HOTKEY.issubset(pressed_vks):
        start_recording()
    # Paste last hotkey (default: Alt+Shift+Z)
    elif vk in PASTE_LAST_HOTKEY and PASTE_LAST_HOTKEY.issubset(pressed_vks) and not _RECORD_ONLY_MODS.intersection(pressed_vks):
        paste_last()


def on_release(key):
    vk = _get_vk(key)
    # Stop recording when the trigger key of the record hotkey is released
    if vk == RECORD_TRIGGER_VK and recording:
        threading.Thread(target=stop_recording, daemon=True).start()
    if vk:
        pressed_vks.discard(vk)


if __name__ == "__main__":
    load_model()
    print("Listening for Ctrl+Shift+F5 (pedal). Ctrl+C to quit.")
    with kb.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
