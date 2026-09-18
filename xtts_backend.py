from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

try:
    import winsound
except ImportError:  # pragma: no cover - keeps structural tests portable.
    winsound = None

from xtts_easy.console import log as console_log
from xtts_easy.projects import update_surface

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
SAMPLES = ROOT / "samples"
DATASETS = ROOT / "datasets"
PROJECTS = ROOT / "projects"
BASE_MODELS = ROOT / "base_models"
RUNTIME = ROOT / ".runtime"
TRAINING = ROOT / "training"
for p in (MODELS, OUTPUTS, SAMPLES, DATASETS, PROJECTS, BASE_MODELS, TRAINING): p.mkdir(parents=True, exist_ok=True)

WHISPER_MODELS = {
    "large-v3 (~10 GB VRAM)": "large-v3",
    "large-v3-turbo (~6 GB VRAM)": "large-v3-turbo",
    "medium (~5 GB VRAM)": "medium",
    "small (~3 GB VRAM)": "small",
}
LANGUAGES = ["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn","ja","hu","ko","hi"]
LANGUAGE_LABELS = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "pt": "Português",
    "pl": "Polski",
    "tr": "Türkçe",
    "ru": "Русский",
    "nl": "Nederlands",
    "cs": "Čeština",
    "ar": "العربية",
    "zh-cn": "中文 (简体)",
    "ja": "日本語",
    "hu": "Magyar",
    "ko": "한국어",
    "hi": "हिन्दी",
}
LANGUAGE_CHOICES = [(LANGUAGE_LABELS[code], code) for code in LANGUAGES]
# Conservative character limits keep text below XTTS' GPT token budget without
# exposing another fragile slider.  The tokenizer, not the UI, remains the
# final authority; these values are only used for automatic chunking/datasets.
XTTS_LANGUAGE_LIMITS = {
    "zh-cn": (1.0, 9.5, 160), "ja": (1.0, 9.5, 160), "ko": (1.0, 9.5, 160),
    "ar": (1.0, 10.5, 190), "hi": (1.0, 10.5, 190),
    "de": (1.0, 11.6, 220), "ru": (1.0, 11.6, 220),
}
CHUNK_CHOICES = ["Automatic (language-aware)", "None", "Sentences", "Paragraphs", "Lines", "Fixed length (language-aware)"]
LEXICON_PRESETS = {
    "Example: Spanish acronyms": {
        "replacements": {"API": "a pe i", "GPU": "ge pe u", "BPE": "be pe e", "GPT": "ge pe te"}
    },
    "Example: English acronyms": {
        "replacements": {"API": "A P I", "GPU": "G P U", "BPE": "B P E", "GPT": "G P T"}
    },
    "Template (empty)": {"replacements": {}}
}
BASE_VERSIONS = ["v2.0.3", "v2.0.2", "v2.0.1", "v2.0.0", "main"]

# These are the optimizer/scheduler families exposed by the training UI.  The
# worker adds the warmup wrapper because GPTTrainerConfig itself has no native
# warmup field.  Prodigy is intentionally opt-in: it is a third-party adaptive
# optimizer, not a scheduler, and uses lr=1 as its documented scale input.
TRAINING_OPTIMIZERS = ["Auto (dataset-aware)", "AdamW", "Adam", "RAdam", "RMSprop", "SGD", "Prodigy"]
TRAINING_SCHEDULERS = [
    "Auto (dataset-aware)",
    "None (optimizer-managed)",
    "MultiStepLR",
    "CosineAnnealingWarmRestarts",
    "CosineAnnealingLR",
    "ExponentialLR",
    "StepLR",
    "ConstantLR",
    "LinearLR",
    "PolynomialLR",
    "OneCycleLR",
    "CyclicLR",
]
TRAINING_SCHEDULER_CADENCES = ["Per optimizer step", "Per epoch"]

_MODEL = None
_MODEL_KEY = None
_ASR = None
_ASR_KEY = None
_CANCEL_AUX = threading.Event()
_TRAIN_PROC = None
_TRAIN_LOG_HANDLE = None
_TRAIN_OUTPUT_THREAD = None
_TRAIN_LOCK = threading.RLock()
_TRAIN_STATE = {
    "running": False,
    "starting": False,
    "status": "Idle",
    "project": "",
    "started": None,
    "returncode": None,
    "log": "",
    "progress_file": "",
    "completion_chimed": False,
}

CHIME_PATH = ROOT / "assets" / "inference_training_done.wav"


def play_done_chime() -> None:
    """Play the non-blocking completion sound used by the Fish-style workflow."""
    try:
        if winsound is not None and CHIME_PATH.is_file():
            winsound.PlaySound(
                str(CHIME_PATH),
                winsound.SND_FILENAME | winsound.SND_ASYNC,
            )
        elif winsound is not None:
            winsound.MessageBeep()
    except Exception as exc:
        # A missing/broken notification must never turn a successful task into
        # a failed Gradio event.
        _training_console(f"Completion chime unavailable: {exc}", level="WARN")


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip()).strip("._") or "item"


def _training_console(message: object, level: str = "INFO") -> None:
    """Send a training lifecycle message to the embedded console and terminal mirror."""
    console_log(f"[TRAIN] {message}", level=level)


def _capture_training_output(process, log_path: Path) -> None:
    """Mirror unbuffered worker output to both the persistent log and Gradio console."""
    try:
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            stream = process.stdout
            if stream is None:
                return
            for raw_line in iter(stream.readline, ""):
                if not raw_line:
                    break
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue
                handle.write(raw_line)
                handle.flush()
                _training_console(line)
    except Exception as exc:
        _training_console(f"Output bridge failed: {exc}", level="ERROR")
    finally:
        try:
            if process.stdout is not None:
                process.stdout.close()
        except Exception:
            pass
        _training_console("Worker output stream closed.")


def normalize_seed(seed_value):
    if seed_value is None or seed_value == "":
        return None
    try:
        seed = int(seed_value)
    except (TypeError, ValueError):
        return None
    if seed == 0:
        return None
    if seed < 0:
        seed = abs(seed)
    return min(seed, 1 << 31)


def random_seed_value():
    return random.randint(1, (1 << 31) - 1)


def apply_generation_seed(seed_value):
    seed = normalize_seed(seed_value)
    if seed is None:
        return None
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return seed


def _path_from_json(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _reference_paths(value) -> list[Path]:
    values = value if isinstance(value, (list, tuple)) else [value]
    paths = []
    for item in values:
        if not item:
            continue
        if isinstance(item, dict):
            item = item.get("path") or item.get("name")
        else:
            item = getattr(item, "path", item)
        if item:
            path = Path(str(item)).expanduser()
            paths.append(path if path.is_absolute() else (ROOT / path).resolve())
    return paths


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _model_artifacts(folder: Path, *, training: bool = False) -> list[Path]:
    names = ["config.json", "model.pth", "vocab.json", "speakers_xtts.pth"]
    if training:
        names += ["dvae.pth", "mel_stats.pth"]
    return [folder / name for name in names]

def unload_all_models():
    global _MODEL, _MODEL_KEY, _ASR, _ASR_KEY
    _MODEL = None; _MODEL_KEY = None; _ASR = None; _ASR_KEY = None
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except Exception:
        pass
    return "Models unloaded and CUDA cache released."

def begin_aux_job(): _CANCEL_AUX.clear()
def stop_aux_job(): _CANCEL_AUX.set(); return "Stop requested."
def _cancelled(): return _CANCEL_AUX.is_set()


def get_sample_choices(reference_mode=None):
    choices = []
    for metadata in SAMPLES.glob("*.json"):
        if reference_mode in ("Single reference mode", "Multiple reference mode"):
            try:
                payload = json.loads(metadata.read_text(encoding="utf-8"))
                is_multiple = (
                    payload.get("reference_mode") == "multiple"
                    or isinstance(payload.get("audio"), list)
                    or (SAMPLES / metadata.stem).is_dir()
                )
                if reference_mode == "Multiple reference mode" and not is_multiple:
                    continue
                if reference_mode == "Single reference mode" and is_multiple:
                    continue
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                if reference_mode == "Multiple reference mode":
                    continue
        choices.append(metadata.stem)
    return sorted(choices, key=str.casefold)

def save_sample(audio_path, name, transcript="", language="en", reference_mode="Single reference mode"):
    source_paths = _reference_paths(audio_path)
    if not source_paths:
        return "Select one or more reference audios.", gr.update()
    if not str(name or "").strip():
        return "Enter a voice name before saving.", gr.update()
    try:
        name = _safe(name)
        multiple = reference_mode == "Multiple reference mode"
        if multiple:
            target_dir = SAMPLES / name
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = SAMPLES
        saved = []
        for index, source_path in enumerate(source_paths if multiple else source_paths[:1], start=1):
            source = _require_file(source_path, "Reference audio")
            info = sf.info(source)
            if info.frames < 1 or info.samplerate < 8000:
                return f"Reference audio is empty or has an invalid sample rate: {source.name}", gr.update()
            suffix = source.suffix.lower() or ".wav"
            if multiple:
                dst = target_dir / f"{index:02d}_{_safe(source.stem)}{suffix}"
            else:
                dst = target_dir / f"{name}{suffix}"
            shutil.copy2(source, dst)
            saved.append(str(dst))
        payload = {
            "schema": 3,
            "name": name,
            "reference_mode": "multiple" if multiple else "single",
            "audio": saved if multiple else saved[0],
            "transcript": (transcript or "").strip(),
            "language": language or "en",
        }
        (SAMPLES / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        count = len(saved)
        label = f"Saved voice package '{name}' ({count} reference{'s' if count != 1 else ''})." if multiple else f"Saved voice '{name}'."
        return label, gr.update(choices=["None", *get_sample_choices()], value=name)
    except Exception as exc:
        return f"Could not save voice: {exc}", gr.update()

def load_sample(name):
    if not name or name == "None": return None, "", "en", "No saved voice selected."
    p = SAMPLES / f"{name}.json"
    if not p.exists(): return None, "", "en", "Saved voice not found."
    try:
        d=json.loads(p.read_text(encoding="utf-8"))
        paths = _reference_paths(d.get("audio"))
        if not paths:
            paths = [SAMPLES / f"{name}.wav"]
        paths = [path for path in paths if path.exists()]
        if not paths:
            return None, d.get("transcript", ""), d.get("language", "en"), f"Audio for '{name}' not found."
        audio = [str(path) for path in paths] if isinstance(d.get("audio"), list) else str(paths[0])
        suffix = f" ({len(paths)} references)" if len(paths) > 1 else ""
        return audio,d.get("transcript",""),d.get("language","en"),f"Loaded '{name}'{suffix}."
    except Exception as exc:
        return None, "", "en", f"Could not load '{name}': {exc}"

def delete_sample(name):
    if not name or name == "None": return "Select a saved voice to delete."
    metadata = SAMPLES / f"{name}.json"
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else {}
        for path in _reference_paths(payload.get("audio")):
            if path.is_file():
                path.unlink()
    except Exception:
        pass
    for ext in (".json",".wav"):
        try:(SAMPLES/f"{name}{ext}").unlink()
        except FileNotFoundError: pass
    package = SAMPLES / name
    if package.is_dir():
        shutil.rmtree(package, ignore_errors=True)
    return f"Deleted '{name}'."


def _load_audio_tensor(path):
    """Decode audio without torchaudio's 2.9 migration warning when possible."""
    try:
        import torch
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(np.asarray(data.T, dtype=np.float32).copy()), int(sample_rate)
    except Exception:
        import torchaudio
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"In 2\.9, this function's implementation will be changed to use torchaudio\.load_with_torchcodec.*",
                category=UserWarning,
                module=r"torchaudio\._backend\.utils",
            )
            return torchaudio.load(str(path))


def _xtts_load_audio(path, sampling_rate):
    """Replacement for Coqui's torchaudio loader used by XTTS references."""
    import torch
    import torchaudio

    audio, source_rate = _load_audio_tensor(path)
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if source_rate != sampling_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sampling_rate)
    audio = audio.clamp(-1, 1)
    return audio


def _load_asr(model_name):
    global _ASR,_ASR_KEY
    key=model_name
    if _ASR is not None and _ASR_KEY==key:return _ASR
    from faster_whisper import WhisperModel
    device="cuda"
    compute_type="float16"
    try:
        import torch
        if not torch.cuda.is_available(): device="cpu"; compute_type="int8"
    except Exception: device="cpu"; compute_type="int8"
    _ASR=WhisperModel(model_name, device=device, compute_type=compute_type, cpu_threads=max(1, min(8, (os.cpu_count() or 4) // 2)))
    _ASR_KEY=key
    return _ASR


def _asr_language(language):
    value = (language or "").strip().lower()
    return {"zh-cn":"zh", "zh_cn":"zh"}.get(value, value)

def transcribe_only(audio_path, model_name="large-v3", language="auto", batch_size=4, progress=gr.Progress()):
    if not audio_path:
        return ""
    try:
        begin_aux_job(); progress(0.05, desc="Loading Faster-Whisper")
        model=_load_asr(model_name)
        segments,info=model.transcribe(
            str(audio_path),
            language=None if language in ("auto","Auto-detect","") else _asr_language(language),
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=False,
        )
        out=[]
        for seg in segments:
            if _cancelled(): break
            text = seg.text.strip()
            if text:
                out.append(text)
        if _cancelled():
            return "Transcription stopped by user."
        progress(1.0, desc="Transcription complete")
        play_done_chime()
        return " ".join(out).strip()
    except Exception as exc:
        return f"[Transcription error] {exc}"


def _model_dir(version): return BASE_MODELS / version

def _ensure_base(version="v2.0.3"):
    if version not in BASE_VERSIONS:
        raise ValueError(f"Unsupported XTTS base version: {version}")
    out=_model_dir(version); out.mkdir(parents=True, exist_ok=True)
    needed=["config.json","model.pth","vocab.json","speakers_xtts.pth","dvae.pth","mel_stats.pth"]
    if all((out/x).exists() for x in needed): return out
    from huggingface_hub import hf_hub_download
    revision=None if version=="main" else version
    for fn in ("config.json","model.pth","vocab.json","speakers_xtts.pth"):
        if not (out/fn).exists():
            hf_hub_download("coqui/XTTS-v2", fn, revision=revision, local_dir=out)
    # The training auxiliaries are published on main and are independent of
    # the versioned inference checkpoint.
    for fn in ("dvae.pth","mel_stats.pth"):
        if not (out/fn).exists():
            hf_hub_download("coqui/XTTS-v2", fn, revision="main", local_dir=out)
    missing=[str(p.name) for p in _model_artifacts(out, training=True) if not p.exists()]
    if missing:
        raise RuntimeError(f"XTTS base download is incomplete ({', '.join(missing)}). Remove the partial folder and retry.")
    return out


def list_ready_models():
    result=[(f"Base XTTS-v2 ({version})", f"base:{version}") for version in BASE_VERSIONS]
    for p in sorted(TRAINING.glob("*/ready")):
        if all(item.exists() for item in _model_artifacts(p)):
            manifest = p / "training_manifest.json"
            label = f"Trained · {p.parent.name}"
            if manifest.exists():
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    language = str(data.get("language") or "").strip()
                    if language:
                        label = f"{label} · {language}"
                except Exception:
                    pass
            result.append((label, str(p)))
    return result

def _resolve_model(choice):
    if not choice or choice.startswith("base:"):
        ver=(choice or "base:v2.0.3").split(":",1)[1]; d=_ensure_base(ver)
    else:
        d=Path(choice).resolve()
        if not d.is_dir():
            raise FileNotFoundError(f"Model folder not found: {d}")
    required = _model_artifacts(d)
    missing = [p.name for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Model is incomplete ({', '.join(missing)}): {d}")
    try:
        from tokenizers import Tokenizer
        vocab_size=Tokenizer.from_file(str(d/"vocab.json")).get_vocab_size(with_added_tokens=True)
        config_data=json.loads((d/"config.json").read_text(encoding="utf-8"))
        configured=[int(value) for value in _nested_values(config_data,{"gpt_number_text_tokens","number_text_tokens"}) if isinstance(value,(int,float)) and int(value)>0]
        if configured and any(value != vocab_size for value in configured):
            raise RuntimeError(f"Tokenizer/config mismatch in {d}: vocab={vocab_size}, config text tokens={configured}")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not validate tokenizer/config for {d}: {exc}") from exc
    return d/"model.pth",d/"config.json",d/"vocab.json",d/"speakers_xtts.pth"

def _load_xtts(choice):
    global _MODEL,_MODEL_KEY
    key=(str(choice),)
    if _MODEL is not None and _MODEL_KEY==key:return _MODEL
    unload_all_models()
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    import TTS.tts.models.xtts as xtts_module
    # Coqui 0.27 still calls torchaudio.load internally for reference audio.
    # Route that call through the local decoder so the torchaudio 2.9 migration
    # warning and its future backend behavior do not leak into inference.
    xtts_module.load_audio = _xtts_load_audio
    ckpt,cfgp,vocab,spk=_resolve_model(choice)
    cfg=XttsConfig(); cfg.load_json(str(cfgp))
    m=Xtts.init_from_config(cfg)
    m.load_checkpoint(cfg, checkpoint_path=str(ckpt), vocab_path=str(vocab), speaker_file_path=str(spk) if spk.exists() else None, use_deepspeed=False)
    if torch.cuda.is_available():m.cuda()
    m.eval(); _MODEL=m; _MODEL_KEY=key
    return m


def _refs(ref_audio, voice_name, library_mode="Multiple reference mode"):
    refs=[]
    if voice_name and voice_name!="None":
        p=SAMPLES/f"{voice_name}.json"
        if p.exists():
            d=json.loads(p.read_text(encoding="utf-8"));
            audio_paths = _reference_paths(d.get("audio")) or [SAMPLES / f"{voice_name}.wav"]
            if library_mode == "Single reference mode":
                audio_paths = audio_paths[:1]
            refs.extend(str(audio) for audio in audio_paths if audio.exists())
    if ref_audio:
        values = ref_audio if isinstance(ref_audio, (list, tuple)) else [ref_audio]
        for value in values:
            if not value:
                continue
            if isinstance(value, dict):
                value = value.get("path") or value.get("name")
            else:
                value = getattr(value, "path", value)
            if value:
                refs.append(str(value))
    return list(dict.fromkeys(refs))

def lexicon_template(preset="Template (empty)"):
    payload = LEXICON_PRESETS.get(preset, LEXICON_PRESETS["Template (empty)"])
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _read_lexicon(value):
    if isinstance(value, dict):
        payload = value
    elif not value:
        return {}
    else:
        raw = str(value).strip()
        try:
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = ROOT / path
            if path.is_file():
                raw = path.read_text(encoding="utf-8")
            payload = json.loads(raw)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
    if not isinstance(payload, dict):
        return {}
    replacements = payload.get("replacements", payload)
    return replacements if isinstance(replacements, dict) else {}


def _automatic_lexicon(model_choice):
    if not model_choice:
        return {}
    candidate = Path(str(model_choice)).expanduser()
    if str(model_choice).startswith("base:"):
        candidate = _model_dir(str(model_choice).split(":", 1)[1])
    path = candidate / "pronunciation_lexicon.json"
    return _read_lexicon(path) if path.is_file() else {}


def apply_lexicon(text, lexicon_path=None, model_choice=None):
    payload = _automatic_lexicon(model_choice)
    payload.update(_read_lexicon(lexicon_path))
    if not payload:
        return text
    # Longest-first replacement prevents a short entry ("san") from
    # modifying a longer pronunciation entry ("santo").  The replacement is
    # intentionally text-only: XTTS still receives ordinary language text and
    # its multilingual tokenizer remains in charge of the final encoding.
    for source, replacement in sorted(payload.items(), key=lambda item: len(str(item[0])), reverse=True):
        source = str(source).strip()
        if not source or str(source).startswith("_"):
            continue
        pattern = rf"(?<![\wÀ-ÿ]){re.escape(source)}(?![\wÀ-ÿ])"
        text = re.sub(pattern, str(replacement), text, flags=re.IGNORECASE)
    return text


def _text_limit(language):
    return XTTS_LANGUAGE_LIMITS.get((language or "en").lower(), (1.0, 11.6, 240))[2]


def _fixed_chunks(text, limit):
    words = text.split()
    chunks, current = [], []
    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > limit:
            chunks.append(" ".join(current).strip())
            current = [word]
        else:
            current.append(word)
    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _split_text(text, mode, language):
    mode = "Automatic (language-aware)" if mode is True else ("None" if mode is False else str(mode or "Automatic (language-aware)"))
    # XTTS' native enable_text_splitting path delegates to spaCy and fails on
    # clean installations without the language-specific spaCy extras. Keep
    # the automatic mode deterministic and dependency-light by splitting here.
    if mode == "Automatic (XTTS)":
        mode = "Automatic (language-aware)"
    if mode == "Automatic (language-aware)":
        mode = "Sentences"
    if mode == "None":
        return [text]
    limit = _text_limit(language)
    if mode == "Paragraphs":
        parts = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    elif mode == "Lines":
        parts = [part.strip() for part in text.splitlines() if part.strip()]
    else:
        parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    chunks = []
    for part in parts or [text]:
        chunks.extend(_fixed_chunks(part, limit) if len(part) > limit else [part])
    return chunks or [text]


def synthesize(model_choice, text, language, voice_name, ref_audio, temperature=.75, top_p=.85, top_k=50, repetition_penalty=10.0, length_penalty=1.0, chunk_mode="Automatic (language-aware)", chunk_gap=.35, lexicon_path="", progress=gr.Progress(), library_mode="Multiple reference mode", seed=None, play_chime=True):
    if not text or not text.strip():return None,"Enter text."
    apply_generation_seed(seed)
    refs=_refs(ref_audio,voice_name,library_mode)
    if not refs:return None,"XTTS requires at least one existing reference audio."
    language = (language or "en").lower()
    if language not in LANGUAGES:
        return None, f"Unsupported XTTS language: {language}"
    text=apply_lexicon(text.strip(),lexicon_path,model_choice)
    try:
        progress(.05,desc="Loading XTTS-v2")
        m=_load_xtts(model_choice)
        progress(.25,desc=f"Computing speaker conditioning from {len(refs)} reference(s)")
        lat,emb=m.get_conditioning_latents(audio_path=refs, gpt_cond_len=m.config.gpt_cond_len, max_ref_length=m.config.max_ref_len, sound_norm_refs=m.config.sound_norm_refs)
        mode = "Automatic (language-aware)" if chunk_mode is True else ("None" if chunk_mode is False else str(chunk_mode or "Automatic (language-aware)"))
        chunks = _split_text(text, mode, language)
        progress(.45,desc=f"Synthesizing {len(chunks)} chunk(s)")
        audio_parts=[]
        for index, chunk in enumerate(chunks):
            out=m.inference(text=chunk, language=language, gpt_cond_latent=lat, speaker_embedding=emb, temperature=float(temperature), length_penalty=float(length_penalty), repetition_penalty=float(repetition_penalty), top_k=int(top_k), top_p=float(top_p), enable_text_splitting=False)
            part=np.asarray(out["wav"],dtype=np.float32)
            if part.size:
                audio_parts.append(part)
            progress(.45 + .45 * ((index + 1) / max(1, len(chunks))), desc=f"Synthesizing chunk {index + 1}/{len(chunks)}")
        if not audio_parts:
            return None, "XTTS returned an empty waveform."
        gap=np.zeros(int(24000 * max(0.0, float(chunk_gap))), dtype=np.float32)
        wav=np.concatenate([piece for index, piece in enumerate(audio_parts) for piece in (([gap] if index else []) + [piece])]) if len(audio_parts) > 1 else audio_parts[0]
        if wav.size == 0:
            return None, "XTTS returned an empty waveform."
        ts=time.strftime("%Y%m%d-%H%M%S")
        path=OUTPUTS/f"xtts-{ts}-{os.getpid()}.wav"
        sf.write(path, wav, 24000)
        progress(1.0, desc="Inference complete")
        if play_chime:
            play_done_chime()
        return str(path),f"Saved: {path.name} · {len(wav)/24000:.2f}s"
    except Exception as exc:
        return None, f"XTTS inference failed: {exc}"


def generate_dialogue(model_choice, language, temperature, top_p, top_k, repetition_penalty, length_penalty, silence, rows, chunk_mode="Automatic (language-aware)", chunk_gap=.35, progress=gr.Progress(), seed=None):
    import soundfile as sf
    apply_generation_seed(seed)
    pieces=[]; sr=24000
    for i,row in enumerate(rows):
        if len(row) == 2:
            voice, text = row
            reference_mode = "Single audio reference"
        else:
            voice, text, reference_mode = row[:3]
        if not text.strip():continue
        library_mode = "Single reference mode" if reference_mode == "Single audio reference" else "Multiple reference mode"
        wav,status=synthesize(model_choice,text,language,voice,None,temperature,top_p,top_k,repetition_penalty,length_penalty,chunk_mode,chunk_gap,"",progress,library_mode=library_mode,play_chime=False)
        if not wav: return None,status
        a,_=sf.read(wav,dtype="float32"); pieces.append(a); pieces.append(np.zeros(int(sr*float(silence)),dtype=np.float32))
    if not pieces:return None,"No dialogue text."
    path=OUTPUTS/f"dialogue-{time.strftime('%Y%m%d-%H%M%S')}.wav"; sf.write(path,np.concatenate(pieces),sr)
    play_done_chime()
    return str(path),f"Saved: {path.name}"


def _audio_files(folder):
    exts={".wav",".mp3",".flac",".m4a",".ogg"}
    return [p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in exts]


def _dataset_limits(language):
    return XTTS_LANGUAGE_LIMITS.get((language or "en").lower(), (1.0, 11.6, 240))


def _smart_eval_fraction(sample_count):
    if sample_count < 12:
        return 0.10
    if sample_count < 50:
        return 0.12
    if sample_count < 200:
        return 0.15
    return 0.18


def _paired_transcript(audio_path):
    for suffix in (".txt", ".lab", ".transcript"):
        candidate = audio_path.with_suffix(suffix)
        if candidate.is_file():
            text = re.sub(r"\s+", " ", candidate.read_text(encoding="utf-8-sig").strip())
            if text:
                return text, candidate
    return "", None


def _paired_text_parts(text, count):
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    if count <= 1:
        return [text]
    if len(sentences) >= count:
        groups = [[] for _ in range(count)]
        for index, sentence in enumerate(sentences):
            groups[min(count - 1, index * count // len(sentences))].append(sentence)
        return [" ".join(group).strip() for group in groups if group]
    return _fixed_chunks(text, max(1, math.ceil(len(text) / count)))[:count]


def _paired_segments(wav, sample_rate, text, min_sec, max_sec):
    duration = wav.shape[-1] / max(1, sample_rate)
    count = max(1, math.ceil(duration / max_sec))
    texts = _paired_text_parts(text, count)
    result = []
    for index, part in enumerate(texts):
        start = int(index * wav.shape[-1] / len(texts))
        end = int((index + 1) * wav.shape[-1] / len(texts))
        part_duration = (end - start) / max(1, sample_rate)
        if part and part_duration >= min_sec:
            result.append((start, end, part))
    return result


def _copy_or_initialize_lexicon(source_folder, output):
    source = Path(source_folder) / "pronunciation_lexicon.json"
    target = output / "pronunciation_lexicon.json"
    if source.is_file():
        payload = _read_lexicon(source)
        target.write_text(json.dumps({"replacements": payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        return "sidecar"
    target.write_text(json.dumps({"replacements": {}}, ensure_ascii=False, indent=2), encoding="utf-8")
    return "automatic-empty"


def prepare_dataset(source_folder, dataset_name, language, whisper_model, eval_percent=None, min_sec=None, max_sec=None, progress=gr.Progress()):
    if not source_folder or not Path(source_folder).exists():return "Source folder not found.",None,None
    if not str(dataset_name or "").strip():return "Enter a dataset name.",None,None
    try:
        import torchaudio
    except Exception as exc:
        return f"torchaudio is not available in the active environment: {exc}",None,None
    language = (language or "en").lower()
    if language not in LANGUAGES:
        return f"Unsupported XTTS language: {language}",None,None
    language = (language or "en").lower()
    min_sec, max_sec, text_max = _dataset_limits(language)
    out=DATASETS/_safe(dataset_name); wavdir=out/"wavs"; wavdir.mkdir(parents=True,exist_ok=True)
    begin_aux_job()
    asr=None; rows=[]; files=_audio_files(source_folder); skipped_long=0; paired_files=0; transcribed_files=0; paired_segments=0
    if not files:return "No supported audio files found.",None,None
    for fi,p in enumerate(files):
        paired_text, paired_file = _paired_transcript(p)
        progress(fi/max(1,len(files)),desc=f"Using {paired_file.name}" if paired_file else f"Transcribing {p.name}")
        try:
            wav,source_sr=_load_audio_tensor(p); wav=wav.mean(0,keepdim=True)
        except Exception as exc:
            return f"Could not decode {p.name}: {exc}",None,None
        if paired_text:
            paired_files += 1
            segments=[type("PairedSegment", (), {"start": start / source_sr, "end": end / source_sr, "text": text}) for start, end, text in _paired_segments(wav, source_sr, paired_text, min_sec, max_sec)]
            paired_segments += len(segments)
        else:
            if asr is None:
                asr=_load_asr(whisper_model)
            transcribed_files += 1
            segments,_=asr.transcribe(str(p),language=_asr_language(language),vad_filter=True,word_timestamps=False,beam_size=5,condition_on_previous_text=False)
        for si,s in enumerate(segments):
            if _cancelled():return "Cancelled.",None,None
            dur=float(s.end-s.start)
            if dur<min_sec or dur>max_sec:continue
            text=re.sub(r"\s+", " ", s.text.strip())
            if not text:continue
            if len(text)>text_max:
                skipped_long += 1
                continue
            a=max(0,int(max(0.0,s.start-.08)*source_sr)); b=min(wav.shape[-1],int((s.end+.08)*source_sr)); clip=wav[:,a:b]
            if clip.shape[-1] < int(min_sec*source_sr):
                continue
            target_sr = 22050
            if source_sr != target_sr:
                clip=torchaudio.functional.resample(clip,source_sr,target_sr)
            target=wavdir/f"{fi:05d}_{p.stem}_{si:05d}.wav"; torchaudio.save(str(target),clip,target_sr,encoding="PCM_S",bits_per_sample=16)
            rows.append((f"wavs/{target.name}",text,"speaker"))
    if len(rows)<4:return f"Only {len(rows)} usable segments were created (minimum is 4).",None,None
    rng=np.random.default_rng(1234); rng.shuffle(rows); eval_fraction=_smart_eval_fraction(len(rows)); n_eval=max(1,min(len(rows)-1,int(len(rows)*eval_fraction))); ev=rows[:n_eval]; tr=rows[n_eval:]
    for fn,data in (("metadata_train.csv",tr),("metadata_eval.csv",ev)):
        with open(out/fn,"w",encoding="utf-8",newline="") as f:
            w=csv.writer(f,delimiter="|"); w.writerow(["audio_file","text","speaker_name"]); w.writerows(data)
    (out/"lang.txt").write_text(language,encoding="utf-8")
    lexicon_mode=_copy_or_initialize_lexicon(source_folder, out)
    manifest={"schema":3,"dataset":_safe(dataset_name),"language":language,"source_files":len(files),"segments":len(rows),"train":len(tr),"eval":len(ev),"eval_fraction":eval_fraction,"audio_min_seconds":min_sec,"audio_max_seconds":max_sec,"text_max_chars":text_max,"skipped_text_over_limit":skipped_long,"paired_files":paired_files,"paired_segments":paired_segments,"transcribed_files":transcribed_files,"sample_rate":22050,"pronunciation_pipeline":"automatic_bpe_hotspot_analysis_and_lexicon","pronunciation_lexicon":lexicon_mode}
    (out/"dataset_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    try:
        update_surface(_safe(dataset_name), "dataset", {"dataset_name":_safe(dataset_name),"source_folder":str(Path(source_folder).resolve()),"language":language,"whisper_model":whisper_model})
    except Exception as exc:
        return f"Prepared {len(tr)} train + {len(ev)} eval segments, but project state could not be saved: {exc}",str(out/"metadata_train.csv"),str(out/"metadata_eval.csv")
    note = f"; skipped {skipped_long} segment(s) over the {text_max}-character language limit" if skipped_long else ""
    pair_note=f"; reused {paired_files} TXT/LAB pair(s) without ASR" if paired_files else "; ASR used only for audio without a sidecar transcript"
    play_done_chime()
    return f"Prepared {len(tr)} train + {len(ev)} eval segments at 22050 Hz{pair_note}{note}.",str(out/"metadata_train.csv"),str(out/"metadata_eval.csv")


def dataset_stats(dataset_name):
    out=DATASETS/_safe(dataset_name); rows=[]
    for fn in ("metadata_train.csv","metadata_eval.csv"):
        p=out/fn
        if p.exists():
            with open(p,encoding="utf-8") as f: rows.extend(list(csv.DictReader(f,delimiter="|")))
    texts=[r.get("text","") for r in rows]; chars=sum(map(len,texts)); words=sum(len(x.split()) for x in texts)
    return f"Samples: {len(rows)} · Words: {words:,} · Characters: {chars:,}"


def _dataset_texts(dataset_name):
    out=DATASETS/_safe(dataset_name); texts=[]
    for fn in ("metadata_train.csv","metadata_eval.csv"):
        p=out/fn
        if p.exists():
            with open(p,encoding="utf-8") as f:
                texts += [re.sub(r"\s+", " ", r.get("text", "").strip()) for r in csv.DictReader(f,delimiter="|")]
    return out, [text for text in texts if text]


def _tokenizer_hotspots(texts, tokenizer):
    from collections import Counter
    counts = Counter()
    details = {}
    for text in texts:
        # Keep accented Unicode letters and apostrophes; XTTS lowercases at
        # inference, so candidates are normalized to lowercase here too.
        for raw in re.findall(r"[\wÀ-ÿ'’-]+", text, flags=re.UNICODE):
            word = raw.strip("'’-_").lower()
            if len(word) < 2 or word.startswith("["):
                continue
            encoded = tokenizer.encode(word)
            tokens = list(encoded.tokens)
            token_ids = list(encoded.ids)
            unk_id = tokenizer.token_to_id("[UNK]")
            if word in tokenizer.get_vocab() or (unk_id is not None and unk_id not in token_ids and len(token_ids) < 3):
                continue
            counts[word] += 1
            details[word] = {"word": word, "pieces": tokens, "token_count": len(token_ids), "unknown": bool(unk_id is not None and unk_id in token_ids)}
    # Unknowns and heavily fragmented words are the best candidates. Frequency
    # breaks ties and avoids spending new embeddings on one-off punctuation.
    ranked = sorted(details.values(), key=lambda item: (
        1 if item["unknown"] else 0,
        item["token_count"],
        counts[item["word"]],
        len(item["word"]),
    ), reverse=True)
    return ranked, counts


def analyze_tokenizer(dataset_name, base_version="v2.0.3"):
    out, texts = _dataset_texts(dataset_name)
    if not texts:return "Dataset metadata not found."
    base=_ensure_base(base_version)
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(base/"vocab.json"))
    corpus="\n".join(texts); enc=tok.encode(corpus)
    unk_id=tok.token_to_id("[UNK]"); unk_count=sum(1 for i in enc.ids if i==unk_id) if unk_id is not None else 0
    hotspots, counts = _tokenizer_hotspots(texts, tok)
    report={"schema":2,"dataset":dataset_name,"base_version":base_version,"token_count":len(enc.ids),"unk_count":unk_count,"unique_words":len(counts),"problem_words":hotspots[:200]}
    rp=out/"tokenizer_analysis.json"; rp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=[f"Corpus tokens: {len(enc.ids):,} · [UNK] tokens: {unk_count} · candidate words: {len(hotspots):,}","Pronunciation/tokenization hotspots (highest priority first):"]+[f"- {item['word']} × {counts[item['word']]}: {item['token_count']} pieces -> {' | '.join(item['pieces'])}{' · [UNK]' if item['unknown'] else ''}" for item in hotspots[:50]]
    return "\n".join(lines)


def build_expanded_tokenizer(dataset_name, base_version="v2.0.3", extra_vocab=256):
    out, texts = _dataset_texts(dataset_name); base=_ensure_base(base_version)
    if not texts:return "Dataset metadata not found.",None
    from tokenizers import Tokenizer, AddedToken
    base_tok=Tokenizer.from_file(str(base/"vocab.json")); old_size=base_tok.get_vocab_size(with_added_tokens=True)
    hotspots, counts = _tokenizer_hotspots(texts, base_tok)
    limit=max(1,min(2048,int(extra_vocab)))
    candidates=[]
    for item in hotspots:
        token = item["word"]
        # Single-word added tokens are matched before the BPE model while all
        # base merges, normalizer and decoder remain untouched. This avoids the
        # incompatible merge-table replacement that caused issue #276.
        if token not in base_tok.get_vocab():
            candidates.append(AddedToken(token, single_word=True, normalized=True))
        if len(candidates)>=limit: break
    if not candidates:
        return "No new pronunciation-sensitive vocabulary was found; keep the stock tokenizer.",None
    expanded_tok=Tokenizer.from_file(str(base/"vocab.json"))
    added_count=expanded_tok.add_tokens(candidates)
    new_size=expanded_tok.get_vocab_size(with_added_tokens=True)
    if added_count != len(candidates) or new_size != old_size + added_count:
        return "Tokenizer expansion did not produce a consistent vocabulary size.",None
    expanded=out/"expanded_vocab.json"; expanded_tok.save(str(expanded))
    manifest={"schema":2,"base_version":base_version,"base_vocab_size":old_size,"expanded_vocab_size":new_size,"added_tokens":[item.content for item in candidates],"strategy":"xtts_bpe_compatible_added_tokens","warning":"Requires the matched resized model.pth before inference or training."}
    (out/"expanded_vocab_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return f"Added {added_count} pronunciation-sensitive BPE entries ({old_size} → {new_size}). Build the matched expanded model next.",str(expanded)


def _set_nested_keys(payload, keys, value):
    if isinstance(payload, dict):
        for key, current in list(payload.items()):
            if key in keys:
                payload[key] = value
            else:
                _set_nested_keys(current, keys, value)
    elif isinstance(payload, list):
        for item in payload:
            _set_nested_keys(item, keys, value)


def _nested_values(payload, keys):
    values=[]
    if isinstance(payload, dict):
        for key, current in payload.items():
            if key in keys:
                values.append(current)
            values.extend(_nested_values(current, keys))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_nested_values(item, keys))
    return values


def expand_model_for_vocab(base_version, expanded_vocab_path, output_dir):
    import torch
    import torch.nn as nn
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from tokenizers import Tokenizer
    base=_ensure_base(base_version); expanded_vocab=Path(expanded_vocab_path).expanduser().resolve(); output=Path(output_dir).resolve(); output.mkdir(parents=True,exist_ok=True)
    if not expanded_vocab.is_file():
        return "Expanded vocab.json was not found.",None
    base_tok=Tokenizer.from_file(str(base/"vocab.json")); new_tok=Tokenizer.from_file(str(expanded_vocab))
    old_token_count=base_tok.get_vocab_size(with_added_tokens=True); new_size=new_tok.get_vocab_size(with_added_tokens=True)
    if new_size <= old_token_count:
        return f"Expanded tokenizer is not larger than the base ({old_token_count}).",None
    # Build the in-memory model from the stock configuration first.  Resizing
    # happens after the base state is loaded; otherwise init_from_config would
    # expect the new rows before it has the old checkpoint to copy from.
    cfg=XttsConfig(); cfg.load_json(str(base/"config.json")); model=Xtts.init_from_config(cfg)
    raw=torch.load(base/"model.pth",map_location="cpu",weights_only=False); state=raw.get("model",raw) if isinstance(raw,dict) else raw
    result=model.load_state_dict(state,strict=False)
    old_size=model.gpt.text_embedding.num_embeddings; dim=model.gpt.text_embedding.embedding_dim
    if old_size != old_token_count:
        return f"Base checkpoint/tokenizer mismatch: model has {old_size} text tokens but vocab has {old_token_count}.",None
    cfg_data=json.loads((base/"config.json").read_text(encoding="utf-8"))
    _set_nested_keys(cfg_data,{"gpt_number_text_tokens","number_text_tokens"},new_size)
    model_args_data=cfg_data.setdefault("model_args",{})
    if not _nested_values(cfg_data,{"gpt_number_text_tokens","number_text_tokens"}):
        model_args_data["gpt_number_text_tokens"]=new_size
    model_args_data["tokenizer_file"]="vocab.json"
    cfg_path=output/"config.json"; cfg_path.write_text(json.dumps(cfg_data,ensure_ascii=False,indent=2),encoding="utf-8")
    emb=nn.Embedding(new_size,dim); head=nn.Linear(dim,new_size)
    with torch.no_grad():
        emb.weight[:old_size].copy_(model.gpt.text_embedding.weight)
        head.weight[:old_size].copy_(model.gpt.text_head.weight)
        head.bias[:old_size].copy_(model.gpt.text_head.bias)
        emb.weight[old_size:].normal_(0,.02); head.weight[old_size:].normal_(0,.02); head.bias[old_size:].zero_()
    model.gpt.text_embedding=emb; model.gpt.text_head=head
    checkpoint=output/"model.pth"; temp=output/"model.pth.tmp"
    torch.save({"model":model.state_dict()},temp); temp.replace(checkpoint)
    shutil.copy2(expanded_vocab,output/"vocab.json"); shutil.copy2(base/"speakers_xtts.pth",output/"speakers_xtts.pth")
    for name in ("dvae.pth","mel_stats.pth"):
        shutil.copy2(base/name,output/name)
    # Re-open the serialized state just enough to prove both text matrices
    # have the same first dimension before the UI advertises the artifact.
    saved=torch.load(checkpoint,map_location="cpu",weights_only=False); saved=saved.get("model",saved)
    if tuple(saved["gpt.text_embedding.weight"].shape)[0] != new_size or tuple(saved["gpt.text_head.weight"].shape)[0] != new_size:
        return "Expanded checkpoint validation failed: text embedding/head dimensions differ.",None
    manifest={"schema":2,"base_version":base_version,"base_vocab_size":old_size,"expanded_vocab_size":new_size,"checkpoint":str(checkpoint),"tokenizer":str(output/"vocab.json"),"missing_base_keys":list(result.missing_keys),"unexpected_base_keys":list(result.unexpected_keys)}
    (output/"expanded_model_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return f"Expanded matched XTTS model from {old_size} to {new_size} text tokens.",str(checkpoint)


def list_datasets():
    return sorted([p.name for p in DATASETS.iterdir() if p.is_dir() and (p/"metadata_train.csv").exists() and (p/"metadata_eval.csv").exists()],key=str.casefold)

def list_checkpoints(project):
    if not project:return ["Fresh / None"]
    root=TRAINING/_safe(project)
    found=sorted(root.rglob("*.pth"),key=lambda p:p.stat().st_mtime,reverse=True)
    return ["Fresh / None", *[str(p) for p in found if p.name in ("best_model.pth","checkpoint.pth") or "checkpoint" in p.name.lower()]]


def _training_scheduler_defaults(train_count, batch_size, grad_accum, epochs, lr):
    """Build step-scaled defaults instead of AllTalk's fixed huge milestones.

    The Coqui trainer steps the scheduler after each optimizer update when
    ``scheduler_after_epoch`` is false.  Scaling against effective optimizer
    updates makes the decay visible in short XTTS runs and still works with
    gradient accumulation.
    """
    effective_batch = max(1, int(batch_size) * int(grad_accum))
    updates_per_epoch = max(1, math.ceil(max(1, int(train_count)) / effective_batch))
    total_updates = max(1, updates_per_epoch * max(1, int(epochs)))
    warmup_steps = max(10, min(500, int(round(total_updates * 0.04))))
    if int(train_count) <= 600:
        scheduler = "CosineAnnealingWarmRestarts"
        params = {
            "T_0": max(25, min(total_updates, updates_per_epoch * 2)),
            "T_mult": 2,
            "eta_min": max(float(lr) * 0.1, 1e-7),
            "last_epoch": -1,
        }
    else:
        scheduler = "MultiStepLR"
        milestones = sorted({
            max(1, int(round(total_updates * 0.55))),
            max(1, int(round(total_updates * 0.78))),
            max(1, int(round(total_updates * 0.92))),
        })
        params = {"milestones": milestones, "gamma": 0.5, "last_epoch": -1}
    return scheduler, warmup_steps, "Per optimizer step", params, updates_per_epoch, total_updates


def autotune_training(dataset_name, base_version="v2.0.3"):
    out=DATASETS/_safe(dataset_name)
    manifest={}
    manifest_path=out/"dataset_manifest.json"
    if manifest_path.is_file():
        try: manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception: manifest={}
    train_count=0
    train_path=out/"metadata_train.csv"
    if train_path.is_file():
        with open(train_path,encoding="utf-8") as handle:
            train_count=max(0,sum(1 for _ in handle)-1)
    try:
        import torch
        vram_gb=(torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0
    except Exception:
        vram_gb=0
    if vram_gb >= 20: batch_size,grad_accum=4,2
    elif vram_gb >= 10: batch_size,grad_accum=2,4
    else: batch_size,grad_accum=1,8
    # XTTS fine-tuning is commonly under-trained when epochs are derived only
    # from a generic small-data heuristic.  These budgets are intentionally
    # longer, while the holdout evaluation and TensorBoard remain the source
    # of truth for stopping or resuming a run.
    if train_count < 24: epochs=60
    elif train_count < 60: epochs=45
    elif train_count < 160: epochs=36
    elif train_count < 400: epochs=28
    elif train_count < 1000: epochs=22
    elif train_count < 3000: epochs=16
    else: epochs=12
    lr=3e-6 if train_count >= 200 else 5e-6
    max_audio_length=float(manifest.get("audio_max_seconds", _dataset_limits(manifest.get("language", "en"))[1]))
    save_step=max(50,min(2000,int(max(1,train_count)*2)))
    eval_steps=max(50,min(2000,int(max(1,train_count))))
    scheduler, warmup_steps, cadence, scheduler_params, updates_per_epoch, total_updates = _training_scheduler_defaults(
        train_count, batch_size, grad_accum, epochs, lr
    )
    optimizer = "AdamW"
    optimizer_params = {"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2}
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "learning_rate": lr,
        "max_audio_length": max_audio_length,
        "save_step": save_step,
        "eval_steps": eval_steps,
        "optimizer": optimizer,
        "optimizer_params": json.dumps(optimizer_params, ensure_ascii=False),
        "scheduler": scheduler,
        "warmup_steps": warmup_steps,
        "scheduler_cadence": cadence,
        "scheduler_params": json.dumps(scheduler_params, ensure_ascii=False),
        "summary": (
            f"Auto-tune: {train_count:,} train samples · {vram_gb:.1f} GB VRAM · batch {batch_size} · "
            f"effective batch {batch_size * grad_accum} · {epochs} epochs · {total_updates:,} optimizer updates · "
            f"{optimizer} + {scheduler} · warmup {warmup_steps} updates · max audio {max_audio_length:.1f}s · "
            f"eval every {eval_steps} steps. The longer budget is intentional; use holdout loss and TensorBoard "
            f"to decide whether a later resume is still improving."
        ),
    }


def prepare_pronunciation_pipeline(dataset_name, base_version, project):
    dataset=DATASETS/_safe(dataset_name)
    training_dir=TRAINING/_safe(project)/"expanded_base"
    lexicon=dataset/"pronunciation_lexicon.json"
    existing_vocab=training_dir/"vocab.json"
    existing_model=training_dir/"model.pth"
    if existing_vocab.is_file() and existing_model.is_file():
        if lexicon.is_file() and not (training_dir/"pronunciation_lexicon.json").exists(): shutil.copy2(lexicon, training_dir/"pronunciation_lexicon.json")
        return True, existing_vocab, existing_model, "Automatic pronunciation assets already available."
    status, expanded_vocab=build_expanded_tokenizer(dataset_name, base_version, 256)
    if not expanded_vocab:
        return False, None, None, status
    status, expanded_model=expand_model_for_vocab(base_version, expanded_vocab, training_dir)
    if not expanded_model:
        return False, None, None, status
    if lexicon.is_file(): shutil.copy2(lexicon, training_dir/"pronunciation_lexicon.json")
    return True, Path(expanded_vocab), Path(expanded_model), f"Automatic pronunciation pipeline: {status}"


def start_training(project,dataset_name,base_version,epochs,batch_size,grad_accum,lr,max_audio_length,save_step,resume_path,training_seed=1234,eval_steps=500,eval_enabled=False,eval_text="",use_expanded_vocab=None,expanded_vocab_path="",expanded_model_path="",checkpoint_mode="disk",optimizer="AdamW",optimizer_params="",scheduler="MultiStepLR",warmup_steps=0,scheduler_cadence="Per optimizer step",scheduler_params=""):
    global _TRAIN_PROC, _TRAIN_LOG_HANDLE, _TRAIN_OUTPUT_THREAD
    if not str(project or "").strip() or str(project).strip() == "None": return "Enter a training project name."
    if not str(dataset_name or "").strip() or str(dataset_name).strip() == "None": return "Select a prepared dataset."
    checkpoint_mode=str(checkpoint_mode or "disk").strip().lower()
    if checkpoint_mode not in {"disk", "ram", "vram"}:
        return "Unsupported checkpoint mode. Choose disk, RAM or VRAM."
    optimizer=str(optimizer or "AdamW").strip()
    scheduler=str(scheduler or "MultiStepLR").strip()
    if optimizer == "Prodigy" and scheduler == "Auto (dataset-aware)":
        # Prodigy's adaptive D-estimate is the schedule.  Do not silently
        # combine it with the dataset-aware cosine/milestone policy.
        scheduler = "None (optimizer-managed)"
        warmup_steps = 0
        scheduler_params = "{}"
    if optimizer == "Auto (dataset-aware)" or scheduler == "Auto (dataset-aware)":
        auto=autotune_training(dataset_name, base_version)
        if optimizer == "Auto (dataset-aware)":
            optimizer=auto["optimizer"]
            optimizer_params=auto["optimizer_params"]
        if scheduler == "Auto (dataset-aware)":
            scheduler=auto["scheduler"]
            warmup_steps=auto["warmup_steps"]
            scheduler_cadence=auto["scheduler_cadence"]
            scheduler_params=auto["scheduler_params"]
    if optimizer == "Prodigy":
        try:
            import prodigyopt  # noqa: F401 - validates the project-local install before spawning the worker.
        except Exception as exc:
            return f"Prodigy is not installed in the project environment. Run install.bat first: {exc}"
        # Prodigy's lr is a scale input, not a conventional XTTS learning rate.
        # Never silently pass a 5e-6 AdamW value to it.
        try:
            requested_lr = float(lr)
        except (TypeError, ValueError):
            requested_lr = 1.0
        if abs(requested_lr - 1.0) > 1e-12:
            _training_console(f"Prodigy requested lr={requested_lr:g}; using documented scale lr=1.0.", level="WARN")
        lr = 1.0
        if scheduler == "Auto (dataset-aware)":
            scheduler = "None (optimizer-managed)"
            warmup_steps = 0
            scheduler_params = "{}"
    if optimizer not in TRAINING_OPTIMIZERS[1:]:
        return f"Unsupported optimizer: {optimizer}."
    if scheduler not in TRAINING_SCHEDULERS[1:]:
        return f"Unsupported scheduler: {scheduler}."
    try:
        parsed_optimizer_params=json.loads(str(optimizer_params or "{}").strip() or "{}")
        parsed_scheduler_params=json.loads(str(scheduler_params or "{}").strip() or "{}")
    except json.JSONDecodeError as exc:
        return f"Optimizer/scheduler parameters must be valid JSON objects: {exc}"
    if not isinstance(parsed_optimizer_params,dict) or not isinstance(parsed_scheduler_params,dict):
        return "Optimizer and scheduler parameters must each be a JSON object."
    scheduler_cadence=str(scheduler_cadence or "Per optimizer step").strip()
    if scheduler_cadence not in TRAINING_SCHEDULER_CADENCES:
        scheduler_cadence="Per optimizer step"
    warmup_steps=max(0,int(warmup_steps or 0))
    project=_safe(project); dataset=DATASETS/_safe(dataset_name); out=TRAINING/project; out.mkdir(parents=True,exist_ok=True)
    _training_console(f"Start requested · project={project} · dataset={dataset.name} · base={base_version} · epochs={int(epochs)} · batch={int(batch_size)} · grad_accum={int(grad_accum)}")
    _training_console(f"Optimization: {optimizer} · scheduler={scheduler} · cadence={scheduler_cadence} · warmup={warmup_steps} updates/epochs")
    _training_console(f"Optimizer params: {json.dumps(parsed_optimizer_params, ensure_ascii=False, sort_keys=True)}")
    _training_console(f"Scheduler params: {json.dumps(parsed_scheduler_params, ensure_ascii=False, sort_keys=True)}")
    _training_console(f"Evaluation: {'prompted audio enabled' if eval_enabled else 'quantitative holdout only'} · every {int(eval_steps)} steps")
    _training_console(f"Checkpoint policy: {checkpoint_mode} · {'one rolling recovery file' if checkpoint_mode == 'disk' else 'no intermediate disk writes'}")
    progress_path = out / "progress.json"
    log_path = out / "training.log"
    with _TRAIN_LOCK:
        if (_TRAIN_PROC and _TRAIN_PROC.poll() is None) or _TRAIN_STATE.get("starting"):
            _training_console("Training is already running or preparing; new request ignored.", level="WARN")
            return "Training is already running."
        _TRAIN_STATE.update(starting=True,status="Preparing training",project=project,started=time.time(),returncode=None,log=str(log_path),progress_file=str(progress_path),completion_chimed=False)
        try:
            progress_path.unlink(missing_ok=True)
        except OSError as exc:
            _training_console(f"Could not clear the previous progress snapshot: {exc}", level="WARN")

    def startup_failure(message):
        with _TRAIN_LOCK:
            _TRAIN_STATE.update(starting=False,running=False,status="Training start failed",returncode=None)
        _training_console(message, level="ERROR")
        return message

    if not (dataset/"metadata_train.csv").exists() or not (dataset/"metadata_eval.csv").exists():
        return startup_failure("Dataset train/eval metadata not found.")
    _training_console("Validated train/eval metadata. Preparing automatic pronunciation assets...")
    automatic_note="Automatic pronunciation pipeline: stock tokenizer retained because no additional BPE entries were required."
    use_expanded=bool(use_expanded_vocab is True)
    if not use_expanded:
        try:
            use_expanded, auto_vocab, auto_model, automatic_note=prepare_pronunciation_pipeline(dataset_name,base_version,project)
            if use_expanded:
                expanded_vocab_path, expanded_model_path = str(auto_vocab), str(auto_model)
            _training_console(automatic_note)
        except Exception as exc:
            return startup_failure(f"Automatic pronunciation preparation failed: {exc}")
    if use_expanded:
        _training_console("Validating matched expanded tokenizer and resized XTTS checkpoint...")
        vocab = Path(str(expanded_vocab_path or "")).expanduser().resolve()
        checkpoint = Path(str(expanded_model_path or "")).expanduser().resolve()
        if not vocab.is_file() or not checkpoint.is_file():
            return startup_failure("Expanded training requires both expanded_vocab.json and the matched model.pth.")
        try:
            import torch
            from tokenizers import Tokenizer
            token_count=Tokenizer.from_file(str(vocab)).get_vocab_size(with_added_tokens=True)
            state=torch.load(checkpoint,map_location="cpu",weights_only=False); state=state.get("model",state)
            if tuple(state["gpt.text_embedding.weight"].shape)[0] != token_count or tuple(state["gpt.text_head.weight"].shape)[0] != token_count:
                return startup_failure("Expanded tokenizer/checkpoint mismatch: text embedding and head must equal the tokenizer size.")
        except Exception as exc:
            return startup_failure(f"Could not validate expanded training pair: {exc}")
    eval_steps=max(0,int(eval_steps))
    eval_text=str(eval_text or "").strip()
    cmd=[sys.executable,str(ROOT/"xtts_easy"/"train_worker.py"),"--project",project,"--dataset",str(dataset),"--base-version",base_version,"--epochs",str(int(epochs)),"--batch-size",str(int(batch_size)),"--grad-accum",str(int(grad_accum)),"--lr",str(float(lr)),"--max-audio-length",str(float(max_audio_length)),"--save-step",str(int(save_step)),"--eval-steps",str(eval_steps),"--training-seed",str(int(training_seed)),"--checkpoint-mode",checkpoint_mode,"--progress-file",str(progress_path),"--optimizer",optimizer,"--optimizer-params",json.dumps(parsed_optimizer_params,ensure_ascii=False),"--scheduler",scheduler,"--warmup-steps",str(warmup_steps),"--scheduler-cadence",scheduler_cadence,"--scheduler-params",json.dumps(parsed_scheduler_params,ensure_ascii=False)]
    if eval_enabled:
        cmd += ["--eval-enabled", "--eval-text", eval_text or "This is a fixed evaluation sample generated during training."]
    if resume_path and resume_path!="Fresh / None":
        resume=Path(str(resume_path)).expanduser().resolve()
        if not resume.is_file():
            return startup_failure(f"Resume checkpoint not found: {resume}")
        cmd += ["--resume",str(resume)]
    if use_expanded:cmd += ["--vocab",str(vocab),"--base-checkpoint",str(checkpoint)]
    with _TRAIN_LOCK:
        if _TRAIN_LOG_HANDLE:
            _TRAIN_LOG_HANDLE.close(); _TRAIN_LOG_HANDLE=None
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START {' '.join(cmd)}\n[{automatic_note}]\n")
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        worker_cmd=[sys.executable,"-u",*cmd[1:]] if cmd and cmd[0] == sys.executable else cmd
        _training_console(f"Launching XTTS worker: {' '.join(worker_cmd)}")
        try:
            _TRAIN_PROC=subprocess.Popen(worker_cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,creationflags=creationflags)
        except Exception as exc:
            _TRAIN_STATE.update(starting=False,running=False,status="Training start failed",returncode=None)
            _training_console(f"Could not launch XTTS worker: {exc}", level="ERROR")
            return f"Could not launch XTTS worker: {exc}"
        _TRAIN_OUTPUT_THREAD=threading.Thread(target=_capture_training_output,args=(_TRAIN_PROC,log_path),name="xtts-training-output",daemon=True)
        _TRAIN_OUTPUT_THREAD.start()
        _TRAIN_STATE.update(running=True,starting=False,status="Training started",project=project,started=time.time(),returncode=None,log=str(log_path),progress_file=str(progress_path),completion_chimed=False)
        _training_console(f"Worker started · PID={_TRAIN_PROC.pid} · log={log_path}")
        try:
            update_surface(project, "training", {"dataset":_safe(dataset_name),"base_version":base_version,"epochs":int(epochs),"batch_size":int(batch_size),"grad_accum":int(grad_accum),"learning_rate":float(lr),"max_audio_length":float(max_audio_length),"save_step":int(save_step),"training_seed":int(training_seed),"checkpoint_mode":checkpoint_mode,"optimizer":optimizer,"optimizer_params":parsed_optimizer_params,"scheduler":scheduler,"warmup_steps":warmup_steps,"scheduler_cadence":scheduler_cadence,"scheduler_params":parsed_scheduler_params,"eval":{"enabled":bool(eval_enabled),"text":eval_text,"every_steps":eval_steps},"resume":str(resume_path or "Fresh / None"),"expanded_tokenizer":bool(use_expanded),"pronunciation_pipeline":"automatic"})
        except Exception as exc:
            _training_console(f"Could not save project state: {exc}", level="WARN")
    return f"Training started for '{project}'. Log: {log_path.name}"

def stop_training():
    global _TRAIN_PROC, _TRAIN_LOG_HANDLE, _TRAIN_OUTPUT_THREAD
    with _TRAIN_LOCK:
        if _TRAIN_STATE.get("starting") and not (_TRAIN_PROC and _TRAIN_PROC.poll() is None):
            _training_console("Stop requested while preparation is still running; the worker is not cancellable until launch.", level="WARN")
            return "Training preparation is still running; stop will be available when the worker starts."
        if _TRAIN_PROC and _TRAIN_PROC.poll() is None:
            _training_console(f"Stop requested for worker PID={_TRAIN_PROC.pid}...", level="WARN")
            _TRAIN_PROC.terminate()
            try: _TRAIN_PROC.wait(timeout=10)
            except subprocess.TimeoutExpired: _TRAIN_PROC.kill()
            _TRAIN_STATE.update(running=False,status="Stop requested",returncode=_TRAIN_PROC.returncode)
            if _TRAIN_LOG_HANDLE:
                _TRAIN_LOG_HANDLE.flush(); _TRAIN_LOG_HANDLE.close(); _TRAIN_LOG_HANDLE=None
            _training_console(f"Worker stopped with exit code {_TRAIN_PROC.returncode}.", level="WARN")
            return "Stop requested."
    _training_console("Stop requested but no training process is running.", level="WARN")
    return "No training process is running."

def training_status():
    global _TRAIN_PROC, _TRAIN_LOG_HANDLE
    with _TRAIN_LOCK:
        running=bool(_TRAIN_PROC and _TRAIN_PROC.poll() is None)
        _TRAIN_STATE["running"]=running
        if not running and _TRAIN_PROC is not None:
            code=_TRAIN_PROC.returncode
            _TRAIN_STATE["returncode"]=code
            if _TRAIN_STATE["status"]=="Training started":
                _TRAIN_STATE["status"]="Training complete" if code==0 else f"Training exited with code {code}"
                if code == 0:
                    _training_console("Training worker completed successfully.", level="INFO")
                    if not _TRAIN_STATE.get("completion_chimed"):
                        play_done_chime()
                        _TRAIN_STATE["completion_chimed"] = True
                else:
                    _training_console(f"Training worker exited with code {code}.", level="ERROR")
            if _TRAIN_LOG_HANDLE:
                _TRAIN_LOG_HANDLE.flush(); _TRAIN_LOG_HANDLE.close(); _TRAIN_LOG_HANDLE=None
        return dict(_TRAIN_STATE)


def training_progress_snapshot():
    state = training_status()
    progress_path = Path(state.get("progress_file") or "")
    payload = {}
    if progress_path.is_file():
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            payload = {}
    running = bool(state.get("running"))
    starting = bool(state.get("starting"))
    code = state.get("returncode")
    total_steps = int(payload.get("total_steps", 0) or 0)
    step = int(payload.get("step", 0) or 0)
    pct = float(payload.get("pct", 0.0) or 0.0)
    if not running and code == 0 and total_steps and step >= total_steps:
        pct = 100.0
    return {
        "running": running,
        "exit_code": code,
        "project": payload.get("project") or state.get("project", ""),
        "status": state.get("status", "Idle"),
        "phase": payload.get("phase") or (state.get("status") if running or starting or state.get("status") not in ("Idle", "Training complete") else ("Training finished" if code is not None else "Idle")),
        "starting": starting,
        "pct": min(100.0, max(0.0, pct)),
        "epoch": int(payload.get("epoch", 0) or 0),
        "total_epochs": int(payload.get("total_epochs", 0) or 0),
        "step": step,
        "total_steps": total_steps,
        "loss": payload.get("loss"),
        "learning_rate": payload.get("learning_rate"),
        "elapsed": payload.get("elapsed", max(0.0, time.time() - float(state.get("started") or time.time())) if running else None),
        "eta": payload.get("eta"),
    }


def tensorboard_logdir(project):
    """Return only the active/latest TensorBoard run for a training project."""
    project_name = _safe(project)
    root = (TRAINING / project_name / "run").resolve()
    if not root.is_dir():
        return None
    state = training_status()
    progress_path = Path(state.get("progress_file") or "")
    active_project = state.get("project") == project_name and bool(state.get("starting") or state.get("running"))
    if state.get("project") == project_name and progress_path.is_file():
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        candidate = Path(str(payload.get("run_dir") or "")).expanduser()
        if candidate.is_dir():
            try:
                candidate.resolve().relative_to(root)
            except ValueError:
                pass
            else:
                return candidate.resolve()
    if active_project:
        # Do not fall back to an older run while a fresh worker is still
        # creating its unique TensorBoard directory.
        return None
    runs = [path for path in root.iterdir() if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime).resolve() if runs else None
