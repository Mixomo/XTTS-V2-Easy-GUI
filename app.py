from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import gradio as gr

import xtts_backend as B
from xtts_easy.console import html_view, log
from xtts_easy.projects import clone_project, delete_surface, list_projects, load_project, save_project, update_surface

APP_TITLE = "XTTS-v2 Easy GUI"
NONE = "None"
MAX_DIALOGUE = 12
_TENSORBOARD_PROC = None
_TENSORBOARD_LOGDIR = ""
_TENSORBOARD_URL = ""
log("[UI] XTTS-v2 Easy GUI module initialized")

CSS = """
html,body,#root{width:100%;min-width:0}
.gradio-container{max-width:none!important;width:100%!important;margin:0 auto!important;padding:18px 28px 28px!important}
.tabs,.tabitem,.gradio-container .block,.gradio-container .form{width:100%!important;max-width:none!important}
.title-section{border-bottom:1px solid var(--border-color-primary);margin-bottom:8px;padding-bottom:8px;align-items:center!important}
.title-section h1,.title-section .prose{margin:0!important;padding:0!important}.title-section button{min-height:36px!important;white-space:nowrap}
.tab-subtitle{opacity:.78;margin:0 0 12px!important}.section-heading{margin:10px 0 2px!important}
.toolbar,.workflow-actions,.project-strip{gap:10px!important;align-items:end!important}.compact{min-width:52px!important;max-width:130px!important}
.seed-layout{gap:14px!important;align-items:stretch!important}.seed-layout>.form{min-width:0!important;flex:1 1 0!important}
.seed-layout>.seed-action{flex:0 0 160px!important;width:160px!important;min-width:160px!important;max-width:160px!important;align-self:center!important}
.seed-action,.seed-action button{min-width:0!important;max-width:100%!important;min-height:74px!important;height:74px!important;white-space:nowrap!important}
.seed-action{display:flex!important;align-items:center!important}.seed-action button{width:100%!important;padding:0 12px!important}
.medium-control{max-width:420px!important}.small-control{max-width:260px!important}
.card{padding:14px 14px 12px!important;margin:6px 0 12px!important;border:1px solid var(--border-color-primary)!important;border-radius:10px!important;background:transparent!important;box-shadow:none!important}
.status-card{padding:10px 12px;border:1px solid var(--border-color-primary);border-radius:9px;background:transparent}
.training-progress-card{padding:10px 12px;border:1px solid var(--border-color-primary);border-radius:9px;background:transparent}
.training-progress-head{display:flex;justify-content:space-between;gap:16px;margin-bottom:8px}
.training-progress-track{width:100%;height:10px;border-radius:999px;background:var(--background-fill-secondary);overflow:hidden}
.training-progress-fill{height:100%;border-radius:999px;background:var(--button-primary-background-fill);transition:width .25s ease}
.training-progress-meta{margin-top:8px;opacity:.82;font-size:.92em}
.console-accordion,.console-accordion>div{border-radius:8px!important}footer{display:none!important}
@media(max-width:900px){.gradio-container{padding:10px!important}.title-section button{min-width:100%!important}.medium-control,.small-control{max-width:none!important}.seed-layout>.seed-action{flex:1 1 0!important;width:auto!important;min-width:0!important;max-width:none!important}}
"""


def voices(reference_mode=None):
    return [NONE, *B.get_sample_choices(reference_mode)]


def models():
    return B.list_ready_models()


def refresh_voices(current=NONE):
    choices = voices()
    return gr.update(choices=choices, value=current if current in choices else NONE)


def refresh_library_voices(mode, current=NONE):
    choices = voices(mode)
    return gr.update(choices=choices, value=current if current in choices else NONE)


def refresh_voice_rows(*current):
    choices = voices()
    return [gr.update(choices=choices, value=value if value in choices else NONE) for value in current]


REFERENCE_MODE_CHOICES = ["Single audio reference", "Multiple audio references"]
LIBRARY_MODE_CHOICES = ["Single reference mode", "Multiple reference mode"]


def _library_mode_for_reference(reference_mode):
    return "Multiple reference mode" if reference_mode == "Multiple audio references" else "Single reference mode"


def dialogue_voice_mode_update(reference_mode, current=NONE):
    choices = voices(_library_mode_for_reference(reference_mode))
    return gr.update(choices=choices, value=current if current in choices else NONE)


def refresh_dialogue_voice_rows(*values):
    modes = list(values[:MAX_DIALOGUE])
    current = list(values[MAX_DIALOGUE:2 * MAX_DIALOGUE])
    return [
        gr.update(
            choices=voices(_library_mode_for_reference(modes[index])),
            value=current[index] if current[index] in voices(_library_mode_for_reference(modes[index])) else NONE,
        )
        for index in range(MAX_DIALOGUE)
    ]


def load_library_voice(name, mode="Single reference mode"):
    audio, _transcript, _language, status = B.load_sample(name)
    paths = [str(path) for path in B._reference_paths(audio)]
    multiple = mode == "Multiple reference mode"
    return (None, paths or None, status) if multiple else (paths[0] if paths else None, None, status)


def save_voice_ui(single_audio, multiple_audio, name, mode):
    selected = multiple_audio if mode == "Multiple reference mode" else single_audio
    message, _ = B.save_sample(selected, name, "", "en", mode)
    saved_name = B._safe(name) if str(message).startswith("Saved ") else NONE
    choices = voices(mode)
    return message, gr.update(choices=choices, value=saved_name if saved_name in choices else NONE)


def library_mode_visibility(mode):
    multiple = mode == "Multiple reference mode"
    return gr.update(visible=not multiple), gr.update(visible=multiple)


def library_mode_update(mode, current=NONE):
    multiple = mode == "Multiple reference mode"
    choices = voices(mode)
    selected = current if current in choices else NONE
    single_value, multiple_value, status = load_library_voice(selected, mode)
    return (
        gr.update(choices=choices, value=selected),
        gr.update(value=single_value, visible=not multiple),
        gr.update(value=multiple_value, visible=multiple),
        status,
    )


def load_voice_preview(name, mode="Single audio reference"):
    audio, _transcript, _language, status = B.load_sample(name)
    paths = [str(path) for path in B._reference_paths(audio)]
    multiple = mode == "Multiple audio references"
    return (None, paths or None, status) if multiple else (paths[0] if paths else None, None, status)


def reference_mode_visibility(mode):
    multiple = mode == "Multiple audio references"
    visible = lambda value: gr.update(visible=value)
    return visible(not multiple), visible(multiple)


def resolve_generation_seed(seed, fixed_seed):
    if fixed_seed:
        return B.normalize_seed(seed) or B.random_seed_value()
    return B.random_seed_value()


def new_seed_pair():
    value = B.random_seed_value()
    return value, value


def run_inference(model_choice, text, language, voice_name, reference_mode, single_ref, multiple_refs, temperature, top_p, top_k, repetition_penalty, length_penalty, chunk_mode, chunk_gap, seed, fixed_seed, lexicon_path):
    selected_refs = multiple_refs if reference_mode == "Multiple audio references" else single_ref
    library_mode = "Single reference mode" if reference_mode == "Single audio reference" else "Multiple reference mode"
    chosen_seed = resolve_generation_seed(seed, fixed_seed)
    audio, status = B.synthesize(model_choice, text, language, voice_name, selected_refs, temperature, top_p, top_k, repetition_penalty, length_penalty, chunk_mode, chunk_gap, lexicon_path, library_mode=library_mode, seed=chosen_seed)
    return audio, status, chosen_seed, chosen_seed


def delete_voice(name, mode="Single reference mode"):
    msg = B.delete_sample(name)
    return gr.update(choices=voices(mode), value=NONE), None, None, msg


def browse_folder(current=""):
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(initialdir=current or str(Path.cwd()))
        root.destroy()
        return path or current
    except Exception:
        return current


def clear_outputs():
    count = 0
    for path in B.OUTPUTS.glob("*"):
        try:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            count += 1
        except Exception:
            pass
    return f"Removed {count} output item(s)."


def status_html(state=None):
    state = state or B.training_status()
    extra = f" · Project: {html.escape(state['project'])}" if state.get("project") else ""
    code = f" · exit {state['returncode']}" if state.get("returncode") is not None else ""
    log_line = f"<br><small>Log: {html.escape(state['log'])}</small>" if state.get("log") else ""
    return f"<b>{html.escape(str(state['status']))}</b>{extra}{code}{log_line}"


def _fmt_duration(seconds):
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def training_progress_html(snapshot=None):
    snap = snapshot or B.training_progress_snapshot()
    pct = min(100.0, max(0.0, float(snap.get("pct", 0.0) or 0.0)))
    running = bool(snap.get("running"))
    starting = bool(snap.get("starting"))
    code = snap.get("exit_code")
    status = str(snap.get("status") or "")
    state = "Preparing" if starting else ("Running" if running else ("Complete" if code == 0 else ("Failed to start" if "failed" in status.lower() else ("Stopped / Failed" if code is not None else "Idle"))))
    phase = str(snap.get("phase") or "Waiting for training")
    project = str(snap.get("project") or "No training project")
    epoch = int(snap.get("epoch", 0) or 0)
    total_epochs = int(snap.get("total_epochs", 0) or 0)
    step = int(snap.get("step", 0) or 0)
    total_steps = int(snap.get("total_steps", 0) or 0)
    loss = snap.get("loss")
    loss_text = "--" if loss is None else f"{float(loss):.5f}"
    learning_rate = snap.get("learning_rate")
    learning_rate_text = "--" if learning_rate is None else f"{float(learning_rate):.3e}"
    optimizer_name = str(snap.get("optimizer") or "")
    adaptive_scale = snap.get("adaptive_scale")
    adaptive_text = ""
    if optimizer_name == "Prodigy":
        d_text = "--" if adaptive_scale is None else f"{float(adaptive_scale):.3e}"
        adaptive_text = f" · Prodigy d {d_text}"
    epoch_text = f"Epoch {epoch}/{total_epochs or '--'}"
    step_text = f"Step {step}/{total_steps or '--'}"
    meta = (f"{html.escape(phase)} · {html.escape(project)} · {epoch_text} · {step_text} · "
            f"Loss {loss_text} · LR {learning_rate_text}{adaptive_text} · Elapsed {_fmt_duration(snap.get('elapsed'))} · ETA {_fmt_duration(snap.get('eta'))}")
    return (f'<div class="training-progress-card"><div class="training-progress-head"><b>{html.escape(state)}</b>'
            f'<span>{pct:.1f}%</span></div><div class="training-progress-track"><div class="training-progress-fill" '
            f'style="width:{pct:.2f}%"></div></div><div class="training-progress-meta">{meta}</div></div>')


def training_poll_ui():
    snapshot = B.training_progress_snapshot()
    running = bool(snapshot.get("running"))
    starting = bool(snapshot.get("starting"))
    start_update = gr.update(value="Preparing..." if starting else ("Training..." if running else "🚀 Start Training"), interactive=not (running or starting))
    stop_update = gr.update(value="🛑 Stop Training", interactive=running)
    return training_progress_html(snapshot), start_update, stop_update


def training_controls_running():
    log("[UI] Training request accepted; preparing the worker...", level="INFO")
    return gr.update(value="Preparing...", interactive=False), gr.update(value="🛑 Stop Training", interactive=False)


def training_controls_idle():
    return gr.update(value="🚀 Start Training", interactive=True), gr.update(value="🛑 Stop Training", interactive=False)


CHECKPOINT_TARGET_CHOICES = ["RAM (recommended)", "VRAM (advanced)"]


def checkpoint_mode_from_ui(use_memory, target):
    if not use_memory:
        return "disk"
    return "vram" if str(target or "").startswith("VRAM") else "ram"


def checkpoint_target_visibility(use_memory):
    return gr.update(visible=bool(use_memory))


def checkpoint_update(project, current="Fresh / None"):
    choices = B.list_checkpoints(project)
    return gr.update(choices=choices, value=current if current in choices else "Fresh / None")


def model_update(current=None):
    choices = models()
    values = [value for _, value in choices]
    return gr.update(choices=choices, value=current if current in values else values[0])


def run_dialogue(model, language, temperature, top_p, top_k, repetition, length, chunk_mode, chunk_gap, seed, fixed_seed, silence, count, *args):
    mode_values = list(args[:MAX_DIALOGUE])
    sample_values = list(args[MAX_DIALOGUE:2 * MAX_DIALOGUE])
    text_values = list(args[2 * MAX_DIALOGUE:3 * MAX_DIALOGUE])
    rows = [
        (sample_values[index], text_values[index], mode_values[index])
        for index in range(min(MAX_DIALOGUE, int(count)))
    ]
    chosen_seed = resolve_generation_seed(seed, fixed_seed)
    audio, status = B.generate_dialogue(model, language, temperature, top_p, top_k, repetition, length, silence, rows, chunk_mode, chunk_gap, seed=chosen_seed)
    return audio, status, chosen_seed, chosen_seed


def start_training_ui(project, dataset, base, epochs, batch, grad, lr, optimizer, optimizer_params, scheduler, warmup_steps, scheduler_cadence, scheduler_params, maxlen, save_step, training_seed, checkpoint_memory, checkpoint_target, eval_steps, eval_enabled, eval_text, resume):
    log(f"[UI] Start Training pressed · project={project or 'None'} · dataset={dataset or 'None'}")
    message = B.start_training(
        project,
        dataset,
        base,
        epochs,
        batch,
        grad,
        lr,
        maxlen,
        save_step,
        resume,
        training_seed=int(training_seed),
        eval_steps=int(eval_steps),
        eval_enabled=bool(eval_enabled),
        eval_text=eval_text or "",
        checkpoint_mode=checkpoint_mode_from_ui(checkpoint_memory, checkpoint_target),
        optimizer=optimizer or "AdamW",
        optimizer_params=optimizer_params or "{}",
        scheduler=scheduler or "MultiStepLR",
        warmup_steps=int(warmup_steps or 0),
        scheduler_cadence=scheduler_cadence or "Per optimizer step",
        scheduler_params=scheduler_params or "{}",
    )
    log(f"[UI] {message}", level="ERROR" if "failed" in str(message).lower() or "not found" in str(message).lower() else "INFO")
    return message


def stop_training_ui():
    message = B.stop_training()
    return message


def dialogue_mutate(action, index, count, *args):
    modes = list(args[:MAX_DIALOGUE])
    samples = list(args[MAX_DIALOGUE:2 * MAX_DIALOGUE])
    texts = list(args[2 * MAX_DIALOGUE:3 * MAX_DIALOGUE])
    count = int(count)
    if action == "insert" and count < MAX_DIALOGUE:
        modes.insert(index + 1, modes[index]); samples.insert(index + 1, samples[index]); texts.insert(index + 1, ""); count += 1
    elif action == "clone" and count < MAX_DIALOGUE:
        modes.insert(index + 1, modes[index]); samples.insert(index + 1, samples[index]); texts.insert(index + 1, texts[index]); count += 1
    elif action == "remove" and count > 1:
        modes.pop(index); samples.pop(index); texts.pop(index); count -= 1
    modes = (modes + [REFERENCE_MODE_CHOICES[0]] * MAX_DIALOGUE)[:MAX_DIALOGUE]
    samples = (samples + [NONE] * MAX_DIALOGUE)[:MAX_DIALOGUE]
    texts = (texts + [""] * MAX_DIALOGUE)[:MAX_DIALOGUE]
    mode_updates = [gr.update(value=modes[index], visible=index < count) for index in range(MAX_DIALOGUE)]
    rows = [
        gr.update(
            choices=voices(_library_mode_for_reference(modes[index])),
            value=samples[index],
            visible=index < count,
        )
        for index in range(MAX_DIALOGUE)
    ]
    text_updates = [gr.update(value=texts[index], visible=index < count) for index in range(MAX_DIALOGUE)]
    row_updates = [gr.update(visible=index < count) for index in range(MAX_DIALOGUE)]
    return [count, *mode_updates, *rows, *text_updates, *row_updates]


def project_update(surface, current=NONE):
    choices = [NONE, *list_projects(surface)]
    return gr.update(choices=choices, value=current if current in choices else NONE)


def sync_dataset_completion(dataset_project_name, current_training_project=NONE, current_dataset=None):
    datasets = B.list_datasets()
    candidate = B._safe(dataset_project_name) if dataset_project_name and dataset_project_name != NONE else NONE
    dataset_ready = candidate in datasets
    dataset_value = candidate if dataset_ready else (current_dataset if current_dataset in datasets else None)
    training_choices = [NONE, *list_projects("training")]
    training_value = candidate if dataset_ready and candidate in training_choices else (current_training_project if current_training_project in training_choices else NONE)
    return (
        gr.update(choices=datasets, value=dataset_value),
        gr.update(choices=training_choices, value=training_value),
    )


def load_dataset_project(name):
    if not name or name == NONE:
        return "", "en", "large-v3 (~10 GB VRAM)"
    data = load_project(name).get("dataset", {})
    whisper = data.get("whisper_model", "large-v3")
    display = next((label for label, value in B.WHISPER_MODELS.items() if value == whisper), whisper)
    return data.get("source_folder", ""), data.get("language", "en"), display


def create_project_ui(name, surface):
    try:
        created = save_project(name)
        return project_update(surface, created), f"Created {surface} project '{created}'."
    except Exception as exc:
        return project_update(surface), f"Could not create project: {exc}"


def clone_project_ui(name, surface):
    if not name or name == NONE:
        return project_update(surface), "Select a project to clone."
    try:
        cloned = clone_project(name)
        return project_update(surface, cloned), f"Cloned '{name}' → '{cloned}'."
    except Exception as exc:
        return project_update(surface, name), f"Could not clone project: {exc}"


def delete_project_ui(name, surface):
    if not name or name == NONE:
        return project_update(surface), "Select a project to delete."
    return project_update(surface), delete_surface(name, surface)


def save_dataset_project_ui(name, source, language, whisper):
    if not name or name == NONE:
        return project_update("dataset"), "Select or create a Dataset project first."
    update_surface(name, "dataset", {"source_folder": source or "", "language": language or "en", "whisper_model": B.WHISPER_MODELS.get(whisper, whisper)})
    return project_update("dataset", name), f"Saved Dataset state for '{name}'. The project name is the dataset name."


def save_training_project_ui(name, dataset, base, epochs, batch, grad, lr, optimizer, optimizer_params, scheduler, warmup_steps, scheduler_cadence, scheduler_params, maxlen, save_step, training_seed, checkpoint_memory, checkpoint_target, eval_steps, eval_enabled, eval_text, resume):
    if not name or name == NONE:
        return project_update("training"), "Select or create a Training project first."
    update_surface(name, "training", {
        "dataset": dataset or "",
        "base_version": base,
        "epochs": int(epochs),
        "batch_size": int(batch),
        "grad_accum": int(grad),
        "learning_rate": float(lr),
        "optimizer": optimizer or "AdamW",
        "optimizer_params": str(optimizer_params or "{}"),
        "scheduler": scheduler or "MultiStepLR",
        "warmup_steps": int(warmup_steps or 0),
        "scheduler_cadence": scheduler_cadence or "Per optimizer step",
        "scheduler_params": str(scheduler_params or "{}"),
        "max_audio_length": float(maxlen),
        "save_step": int(save_step),
        "training_seed": int(training_seed),
        "checkpoint_mode": checkpoint_mode_from_ui(checkpoint_memory, checkpoint_target),
        "eval": {
            "enabled": bool(eval_enabled),
            "text": str(eval_text or "").strip(),
            "every_steps": int(eval_steps),
        },
        "resume": resume or "Fresh / None",
        "pronunciation_pipeline": "automatic",
    })
    return project_update("training", name), f"Saved Training state for '{name}'."


def load_training_project(name, current_resume="Fresh / None"):
    if not name or name == NONE:
        return gr.update(choices=B.list_datasets(), value=None), "v2.0.3", 10, 2, 4, 5e-6, "AdamW", '{"betas":[0.9,0.96],"eps":1e-8,"weight_decay":0.01}', "MultiStepLR", 0, "Per optimizer step", '{}', 11.6, 500, 1234, False, gr.update(value="RAM (recommended)", visible=False), 500, False, "This is a fixed evaluation sample generated during training.", gr.update(visible=False), gr.update(choices=["Fresh / None"], value="Fresh / None")
    data = load_project(name).get("training", {})
    evaluation = data.get("eval", {}) or {}
    eval_enabled = bool(evaluation.get("enabled", data.get("eval_enabled", False)))
    eval_text = str(evaluation.get("text", data.get("eval_text", "This is a fixed evaluation sample generated during training.")) or "")
    eval_steps = int(evaluation.get("every_steps", data.get("eval_steps", 500)) or 500)
    checkpoint_mode = str(data.get("checkpoint_mode", "disk") or "disk").lower()
    checkpoint_memory = checkpoint_mode in {"ram", "vram"}
    checkpoint_target = "VRAM (advanced)" if checkpoint_mode == "vram" else "RAM (recommended)"
    optimizer = data.get("optimizer", "AdamW")
    if optimizer not in B.TRAINING_OPTIMIZERS:
        optimizer = "AdamW"
    scheduler = data.get("scheduler", "MultiStepLR")
    if scheduler not in B.TRAINING_SCHEDULERS:
        scheduler = "MultiStepLR"
    optimizer_params = data.get("optimizer_params", {})
    scheduler_params = data.get("scheduler_params", {})
    optimizer_params = json.dumps(optimizer_params, ensure_ascii=False) if isinstance(optimizer_params, dict) else str(optimizer_params or "{}")
    scheduler_params = json.dumps(scheduler_params, ensure_ascii=False) if isinstance(scheduler_params, dict) else str(scheduler_params or "{}")
    resume = data.get("resume", "Fresh / None")
    choices = B.list_checkpoints(name)
    if resume not in choices:
        choices.append(resume)
    datasets = B.list_datasets()
    selected = data.get("dataset") if data.get("dataset") in datasets else None
    return gr.update(choices=datasets, value=selected), data.get("base_version", "v2.0.3"), data.get("epochs", 10), data.get("batch_size", 2), data.get("grad_accum", 4), data.get("learning_rate", 5e-6), optimizer, optimizer_params, scheduler, data.get("warmup_steps", 0), data.get("scheduler_cadence", "Per optimizer step"), scheduler_params, data.get("max_audio_length", 11.6), data.get("save_step", 500), data.get("training_seed", 1234), checkpoint_memory, gr.update(value=checkpoint_target, visible=checkpoint_memory), eval_steps, eval_enabled, eval_text, gr.update(visible=eval_enabled), gr.update(choices=choices, value=resume)


def prepare_dataset_ui(source, project, language, whisper):
    if not project or project == NONE:
        return "Select or create a Dataset Project first; its name is used as the dataset name.", None, None, project_update("dataset")
    result = B.prepare_dataset(source, project, language, B.WHISPER_MODELS[whisper])
    return (*result, project_update("dataset", B._safe(project)))


def autotune_ui(dataset, base):
    if not dataset:
        return 10, 2, 4, 5e-6, "AdamW", '{"betas":[0.9,0.96],"eps":1e-8,"weight_decay":0.01}', "MultiStepLR", 0, "Per optimizer step", '{}', 11.6, 500, 500, "Select a prepared dataset before using Auto-tune."
    try:
        config = B.autotune_training(dataset, base)
        return config["epochs"], config["batch_size"], config["grad_accum"], config["learning_rate"], config["optimizer"], config["optimizer_params"], config["scheduler"], config["warmup_steps"], config["scheduler_cadence"], config["scheduler_params"], config["max_audio_length"], config["save_step"], config["eval_steps"], config["summary"]
    except Exception as exc:
        return 10, 2, 4, 5e-6, "AdamW", '{"betas":[0.9,0.96],"eps":1e-8,"weight_decay":0.01}', "MultiStepLR", 0, "Per optimizer step", '{}', 11.6, 500, 500, f"Auto-tune failed: {exc}"


def optimizer_selection_ui(name, current_lr, current_params, current_scheduler, current_warmup, current_scheduler_params, current_cadence):
    """Apply safe UI defaults when switching into or out of Prodigy."""
    prodigy_params = '{"betas":[0.9,0.999],"eps":1e-8,"weight_decay":0.0,"decouple":true,"use_bias_correction":false,"safeguard_warmup":true,"d0":1e-6,"d_coef":0.5,"slice_p":11}'
    adamw_params = '{"betas":[0.9,0.96],"eps":1e-8,"weight_decay":0.01}'
    if name == "Prodigy":
        return (
            gr.update(value=1.0),
            gr.update(value=prodigy_params),
            gr.update(value="None (optimizer-managed)"),
            gr.update(value=0),
            gr.update(value="{}"),
            gr.update(value="Per optimizer step"),
        )
    try:
        current_lr_float = float(current_lr)
    except (TypeError, ValueError):
        current_lr_float = None
    if current_lr_float == 1.0 and str(current_scheduler or "") == "None (optimizer-managed)" and "d_coef" in str(current_params or ""):
        return (
            gr.update(value=5e-6),
            gr.update(value=adamw_params),
            gr.update(value="MultiStepLR"),
            gr.update(value=0),
            gr.update(value="{}"),
            gr.update(value=current_cadence or "Per optimizer step"),
        )
    return tuple(gr.update() for _ in range(6))


with gr.Blocks(title=APP_TITLE, theme=gr.themes.Default(), css=CSS) as app:
    with gr.Row(elem_classes="title-section"):
        with gr.Column(scale=7):
            gr.Markdown("# 🎙️ XTTS-v2 Easy GUI")
            gr.Markdown("Coqui XTTS-v2 · reusable Voice Library · reference-faithful inference · smart dataset preparation · automatic pronunciation-aware fine-tuning", elem_classes="tab-subtitle")
        with gr.Column(scale=3):
            with gr.Row():
                unload = gr.Button("🧹 Unload Models / Free VRAM", variant="secondary")
                clear = gr.Button("🗑️ Clear Outputs", variant="secondary")
    hidden = gr.Markdown(visible=False)
    unload.click(B.unload_all_models, outputs=hidden, queue=False)
    clear.click(clear_outputs, outputs=hidden, queue=False)

    with gr.Tabs():
        with gr.Tab("🎙️ Prep Samples"):
            gr.Markdown("*Create reusable XTTS speaker references. The same Voice Library is available in Single Inference and Dialogue Builder.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""1. Choose **Single reference mode** for one clip or **Multiple reference mode** to build a reusable package of clean clips. The **Voice Library** dropdown is filtered to the active mode, so individual references and packages cannot be mixed accidentally.
2. Give the entry a stable **Voice Name** and press **Save Voice**. A multiple-mode entry stores every selected audio file as one Voice Library package.
3. **↻ Refresh** rescans the library after files were added externally. **🗑️** removes only the selected saved voice and its package files.
4. For best cloning consistency, use representative clips with natural prosody rather than heavily processed recordings.""")
            with gr.Row():
                with gr.Column():
                    voice_mode = gr.Dropdown(LIBRARY_MODE_CHOICES, value="Single reference mode", label="Voice Library Reference Mode", info="Single saves one reference clip. Multiple saves a reusable package of several reference clips.")
                    with gr.Row():
                        voice = gr.Dropdown(voices("Single reference mode"), value=NONE, label="Voice Library", info="Reusable speaker references filtered by the selected library mode.")
                        vref = gr.Button("↻", elem_classes="compact")
                        vdel = gr.Button("🗑️", variant="stop", elem_classes="compact")
                    audio = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Reference Audio", visible=True)
                    multiple_audio = gr.File(file_count="multiple", file_types=["audio"], type="filepath", label="Reference Audio Package (select or drag and drop)", visible=False)
                with gr.Column():
                    name = gr.Textbox(label="Voice Name", placeholder="speaker_name", info="Stable library identifier; letters, numbers, dots, underscores and hyphens are kept.")
                    save = gr.Button("Save Voice", variant="primary")
                    vstatus = gr.Markdown("No saved voice selected.")
            voice_mode.change(library_mode_update, [voice_mode, voice], [voice, audio, multiple_audio, vstatus], queue=False)
            voice.change(load_library_voice, [voice, voice_mode], [audio, multiple_audio, vstatus], queue=False)
            vref.click(library_mode_update, [voice_mode, voice], [voice, audio, multiple_audio, vstatus], queue=False)
            vdel.click(delete_voice, [voice, voice_mode], [voice, audio, multiple_audio, vstatus], queue=False)
            save.click(save_voice_ui, [audio, multiple_audio, name, voice_mode], [vstatus, voice], queue=False)

        with gr.Tab("🔊 Inference"):
            gr.Markdown("*Use the stock XTTS-v2 base or any completed fine-tune through the same voice, pronunciation and chunking workflow.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""**Model** selects the stock base or a finished fine-tune. **Voice Library** chooses the speaker reference. Its **Audio Reference Mode** is separate from the library's save mode: Single uses one clip, while Multiple consumes every clip in a saved package or uploaded file list.

XTTS fine-tuning produces a complete compatible checkpoint, so trained models are listed with the `Trained ·` prefix and are selected directly. This GUI does not load PEFT/LoRA adapters.

**Temperature** controls variation, **Top-P/Top-K** constrain sampling, **Repetition Penalty** discourages loops, and **Length Penalty** influences duration. Start from the defaults and change one control at a time.

**Split / Chunking Rule** controls long-form handling. `Automatic (language-aware)` performs dependency-free sentence splitting, `None` makes one uninterrupted request, and the explicit sentence/paragraph/line/fixed modes synthesize chunks separately and join them with **Chunk Gap** silence. This avoids XTTS' optional spaCy dependency while keeping the same workflow.""")
            with gr.Row(elem_classes="toolbar"):
                model = gr.Dropdown(models(), value="base:v2.0.3", label="Model", info="Base XTTS or a validated completed fine-tune.", scale=4)
                mref = gr.Button("↻", elem_classes="compact")
                inf_lang = gr.Dropdown(B.LANGUAGE_CHOICES, value="en", label="Language", info="Human-readable language name; XTTS receives the internal code.", scale=2)
            with gr.Accordion("🎛️ Generation Settings", open=True):
                with gr.Row():
                    temp = gr.Slider(.1, 1.5, .75, .05, label="Temperature", info="Sampling variation. Lower is steadier; higher is more expressive but less deterministic.")
                    top_p = gr.Slider(.1, 1, .85, .05, label="Top-P", info="Nucleus sampling cutoff; lower values restrict the candidate distribution.")
                    top_k = gr.Slider(1, 100, 50, 1, label="Top-K", info="Maximum candidate tokens considered at each step.")
                    rep = gr.Slider(1, 20, 10, .5, label="Repetition Penalty", info="Discourages repeated tokens and looping; extreme values can damage prosody.")
                with gr.Row():
                    length = gr.Slider(.1, 3, 1, .1, label="Length Penalty", info="Biases generated duration; keep near 1.0 unless the result is consistently too short or long.")
                    chunk_mode = gr.Dropdown(B.CHUNK_CHOICES, value="Automatic (language-aware)", label="Split / Chunking Rule", info="Dependency-free automatic, uninterrupted, or explicit language-aware chunking for long text.")
                    chunk_gap = gr.Slider(0, 2, .35, .05, label="Chunk Gap (s)", info="Silence inserted between long-form chunks; it does not affect Dialogue Silence.")
                with gr.Row(elem_classes="seed-layout"):
                    seed = gr.Number(0, precision=0, minimum=0, maximum=(1 << 31) - 1, label="Seed", info="Seed used by Python, NumPy and PyTorch before XTTS generation. Zero means a new random seed.", scale=4)
                    fixed_seed = gr.Checkbox(False, label="Enable Fixed Seed", info="OFF = new random seed for every generation. ON = reuse the visible seed for reproducible output.", scale=4)
                    last_seed = gr.State(0)
                    reuse_seed = gr.Button("Reuse Last Seed", variant="secondary", size="lg", elem_classes="seed-action", scale=0, min_width=160)
                    random_seed = gr.Button("Random Seed", variant="secondary", size="lg", elem_classes="seed-action", scale=0, min_width=160)
            with gr.Tabs():
                with gr.Tab("Single Inference"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            reference_mode = gr.Dropdown(REFERENCE_MODE_CHOICES, value="Single audio reference", label="Audio Reference Mode", info="Choose one reference clip or upload several clips; multiple mode uses Gradio's native multi-file selector and drag-and-drop area.")
                            with gr.Row():
                                inf_voice = gr.Dropdown(voices(), value=NONE, label="Voice Library", info="Select a reusable speaker reference; selecting it loads its waveform below.", scale=8)
                                inf_voice_refresh = gr.Button("↻", elem_classes="compact")
                            ref = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Reference Audio / Voice Preview")
                            multiple_refs = gr.File(file_count="multiple", file_types=["audio"], type="filepath", label="Multiple Audio References (select or drag and drop)", visible=False)
                            inf_voice_status = gr.Markdown("No saved voice selected.")
                        with gr.Column(scale=2):
                            text = gr.Textbox(label="Text", lines=8, info="Text to synthesize. Chunking and the automatic pronunciation pipeline are applied before inference.")
                            with gr.Accordion("📚 Pronunciation Lexicon (optional)", open=False):
                                gr.Markdown("""Use a lexicon only for words whose written form is pronounced incorrectly, such as acronyms, names or technical terms. The key is the text XTTS sees; the value is the pronunciation-friendly text that is substituted before tokenization.

Example:
```json
{"replacements": {"API": "a pe i", "XTTS": "ex te te es"}}
```

Leave the editor empty when no custom replacement is needed; the automatic pronunciation pipeline still runs internally.""")
                                lexicon = gr.Textbox(label="Pronunciation Lexicon JSON (optional)", lines=6, placeholder='{"replacements": {"API": "a pe i"}}', info="Paste JSON or a file path. Keys are written forms; values are the spoken approximation. Leave empty for the automatic model-side pipeline.")
                                with gr.Row():
                                    lexicon_preset = gr.Dropdown(list(B.LEXICON_PRESETS), value="Template (empty)", label="Lexicon Example / Preset", info="Loads a safe illustrative replacement map into the editor.")
                                    lexicon_example = gr.Button("Insert Example", elem_classes="compact")
                            with gr.Row(elem_classes="workflow-actions"):
                                generate = gr.Button("Generate Audio 🚀", variant="primary", size="lg")
                            out_audio = gr.Audio(type="filepath", label="Generated Audio")
                            inf_status = gr.Markdown()
                    inf_voice.change(load_voice_preview, [inf_voice, reference_mode], [ref, multiple_refs, inf_voice_status], queue=False)
                    inf_voice_refresh.click(refresh_voices, inf_voice, inf_voice, queue=False)
                    inf_voice_refresh.click(load_voice_preview, [inf_voice, reference_mode], [ref, multiple_refs, inf_voice_status], queue=False)
                    reference_mode.change(reference_mode_visibility, reference_mode, [ref, multiple_refs], queue=False)
                    reference_mode.change(load_voice_preview, [inf_voice, reference_mode], [ref, multiple_refs, inf_voice_status], queue=False)
                    generate.click(run_inference, [model, text, inf_lang, inf_voice, reference_mode, ref, multiple_refs, temp, top_p, top_k, rep, length, chunk_mode, chunk_gap, seed, fixed_seed, lexicon], [out_audio, inf_status, seed, last_seed])
                    lexicon_example.click(B.lexicon_template, lexicon_preset, lexicon, queue=False)
                with gr.Tab("Dialogue Builder"):
                    gr.Markdown("*Each visible row is one speaker turn. Insert, clone or remove rows directly; the count is managed by the row workflow, like the Fish reference GUI.*", elem_classes="tab-subtitle")
                    with gr.Accordion("📖 Quick Guide", open=False):
                        gr.Markdown("""Choose a Voice Library reference and write one turn per row. **Insert** creates a new blank turn after that row, **Clone** duplicates its voice and text, and **Remove** deletes it. The hidden row count is maintained automatically, so there is no redundant manual turn-count slider.

The shared Generation Settings above apply to every turn. **Chunk Gap** is used inside a long turn; **Dialogue Silence** is inserted between speaker turns. Use **↻ Refresh Voice Library** after adding or deleting voices outside this page.""")
                    with gr.Row(elem_classes="toolbar"):
                        silence = gr.Slider(0, 2, .35, .05, label="Dialogue Silence (s)", info="Silence inserted between speakers; independent from long-text Chunk Gap.")
                        dialogue_voice_refresh = gr.Button("↻ Refresh Voice Library", variant="secondary")
                    dialogue_count = gr.State(2)
                    dmodes, dvoices, dtexts, drows, dinsert, dclone, dremove = [], [], [], [], [], [], []
                    for index in range(MAX_DIALOGUE):
                        with gr.Row(visible=index < 2, elem_classes="card") as row:
                            dm = gr.Dropdown(REFERENCE_MODE_CHOICES, value="Single audio reference", label="Audio Reference Mode", info="Single uses the first clip in the selected package; Multiple uses every clip in that Voice Library package.", scale=2)
                            dv = gr.Dropdown(voices("Single reference mode"), value=NONE, label=f"Turn {index + 1} Voice", info="Speaker reference filtered by this row's audio reference mode.", scale=2)
                            dt = gr.Textbox(label="Text", lines=2, info="Only this speaker's line; leave unused hidden rows untouched.", scale=6)
                            with gr.Row(elem_classes="workflow-actions"):
                                ins = gr.Button("＋ Insert", size="sm"); cln = gr.Button("⧉ Clone", size="sm"); rem = gr.Button("− Remove", size="sm")
                        dmodes.append(dm); dvoices.append(dv); dtexts.append(dt); drows.append(row); dinsert.append(ins); dclone.append(cln); dremove.append(rem)
                        dm.change(dialogue_voice_mode_update, [dm, dv], dv, queue=False)
                    dialogue_inputs = [dialogue_count, *dmodes, *dvoices, *dtexts]
                    dialogue_outputs = [dialogue_count, *dmodes, *dvoices, *dtexts, *drows]
                    for index in range(MAX_DIALOGUE):
                        dinsert[index].click(lambda count, *values, _index=index: dialogue_mutate("insert", _index, count, *values), dialogue_inputs, dialogue_outputs, queue=False)
                        dclone[index].click(lambda count, *values, _index=index: dialogue_mutate("clone", _index, count, *values), dialogue_inputs, dialogue_outputs, queue=False)
                        dremove[index].click(lambda count, *values, _index=index: dialogue_mutate("remove", _index, count, *values), dialogue_inputs, dialogue_outputs, queue=False)
                    dgen = gr.Button("Generate Dialogue 🚀", variant="primary", size="lg"); dout = gr.Audio(type="filepath", label="Dialogue Output"); dstatus = gr.Markdown()
                    dgen.click(run_dialogue, [model, inf_lang, temp, top_p, top_k, rep, length, chunk_mode, chunk_gap, seed, fixed_seed, silence, dialogue_count, *dmodes, *dvoices, *dtexts], [dout, dstatus, seed, last_seed])
                    dialogue_voice_refresh.click(refresh_dialogue_voice_rows, [*dmodes, *dvoices], dvoices, queue=False)
            mref.click(model_update, model, model, queue=False)
            reuse_seed.click(lambda value: value or B.random_seed_value(), last_seed, seed, queue=False)
            random_seed.click(new_seed_pair, outputs=[seed, last_seed], queue=False)

        with gr.Tab("📂 Dataset Preparation"):
            gr.Markdown("*Prepare a Coqui-format XTTS dataset with sidecar reuse, language-aware limits and automatic pronunciation metadata.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""1. Create or select a **Dataset Project**. The project name is the dataset name; there is no second redundant field.
2. Select the source folder. For every `clip.wav`, `clip.txt`, `clip.lab` or `clip.transcript` is preferred and the audio is **not transcribed**. ASR is loaded only for audio without a usable sidecar.
3. Choose the language by name and an optional Faster-Whisper model. Internal minimum/maximum audio duration, text limit and eval fraction are selected from the language and dataset size.
4. Build the dataset. Existing metadata is replaced with the new manifest, which records paired reuse, ASR usage, limits and pronunciation pipeline state.
5. The generated `pronunciation_lexicon.json` is carried into automatic training and finished-model inference. Put a source-sidecar lexicon in the source folder when specific names/acronyms need custom readings.""")
            with gr.Row(elem_classes="project-strip"):
                dataset_project = gr.Dropdown([NONE, *list_projects("dataset")], value=NONE, label="Dataset Project", info="This project name is also used as the dataset name.", scale=3)
                dataset_new = gr.Textbox(label="New Project Name", placeholder="my_voice", scale=2)
                dataset_create = gr.Button("Create Project"); dataset_save = gr.Button("💾 Save Project"); dataset_clone = gr.Button("Clone Project", variant="secondary"); dataset_refresh = gr.Button("↻", elem_classes="compact"); dataset_delete = gr.Button("🗑️", variant="stop", elem_classes="compact")
            dataset_project_status = gr.Markdown(elem_classes="status-card")
            with gr.Row():
                source = gr.Textbox(label="Source Audio Folder", info="Folder containing audio and optional same-stem TXT/LAB/transcript sidecars.", scale=6)
                browse = gr.Button("📁 Browse", elem_classes="compact")
                ds_lang = gr.Dropdown(B.LANGUAGE_CHOICES, value="en", label="Language", info="Human-readable name; determines internal audio/text safety limits.", scale=2)
            with gr.Row():
                ds_wm = gr.Dropdown(list(B.WHISPER_MODELS), value="large-v3 (~10 GB VRAM)", label="Whisper Model", info="Used only for files without a valid transcript sidecar.", scale=3)
            with gr.Row(elem_classes="workflow-actions"):
                build = gr.Button("🧱 Build Dataset", variant="primary"); cancel_dataset = gr.Button("🛑 Cancel", variant="stop", elem_classes="compact")
            ds_status = gr.Markdown(elem_classes="status-card"); train_csv = gr.Textbox(label="Train CSV", interactive=False); eval_csv = gr.Textbox(label="Eval CSV", interactive=False)
            browse.click(browse_folder, source, source, queue=False)
            dataset_build_event = build.click(prepare_dataset_ui, [source, dataset_project, ds_lang, ds_wm], [ds_status, train_csv, eval_csv, dataset_project])
            cancel_dataset.click(B.stop_aux_job, outputs=ds_status, queue=False)
            dataset_project.change(load_dataset_project, dataset_project, [source, ds_lang, ds_wm], queue=False)
            dataset_create.click(lambda name: create_project_ui(name, "dataset"), dataset_new, [dataset_project, dataset_project_status], queue=False)
            dataset_save.click(save_dataset_project_ui, [dataset_project, source, ds_lang, ds_wm], [dataset_project, dataset_project_status], queue=False)
            dataset_clone.click(lambda name: clone_project_ui(name, "dataset"), dataset_project, [dataset_project, dataset_project_status], queue=False)
            dataset_refresh.click(lambda: project_update("dataset"), outputs=dataset_project, queue=False)
            dataset_delete.click(lambda name: delete_project_ui(name, "dataset"), dataset_project, [dataset_project, dataset_project_status], queue=False)

        with gr.Tab("🚀 Fine-tuning"):
            gr.Markdown("*Fine-tune the XTTS GPT encoder with resume support, intelligent Auto-tune and an automatic pronunciation-aware tokenizer pipeline.*", elem_classes="tab-subtitle")
            with gr.Accordion("📖 Quick Guide", open=False):
                gr.Markdown("""1. Create or select a Training Project and choose a prepared dataset.
2. Press **⚙ Auto-tune Training** after selecting the dataset. It uses dataset size, language-aware limits and available VRAM to set a longer convergence budget, effective batch size, learning rate, scheduler, warmup and checkpoint cadence.
3. The pronunciation pipeline runs internally for every training project: it analyzes BPE hotspots, adds compatible whole-word tokens only when justified, resizes the XTTS embedding/head pair together, and carries the model lexicon into the finished model. There is no separate tokenizer tab or manual mismatch-prone checkbox.
4. **Evaluation Zone** is optional prompted monitoring. XTTS always evaluates the holdout metadata; when enabled, the same fixed sentence is synthesized with a deterministic reference from that holdout and logged to TensorBoard at each epoch. **Evaluate Every (Steps)** controls the quantitative holdout validation cadence.
5. The scheduler controls mirror the installed Coqui/PyTorch trainer: **MultiStepLR** is the AllTalk-compatible milestone decay, **CosineAnnealingWarmRestarts** periodically lowers and restarts the rate, **CosineAnnealingLR** decays smoothly once, **StepLR/ExponentialLR** provide classic decay, and **OneCycleLR/CyclicLR** are advanced per-update policies. **Warmup Steps** ramps from a small rate to the requested learning rate before the selected schedule.
6. **Prodigy** is an experimental adaptive optimizer, not a scheduler. It intentionally changes the learning-rate field to `1.0`, uses **None (optimizer-managed)** by default, and reports the effective adaptive LR and D estimate in the console/progress card. Its default `slice_p=11` reduces adaptation-state memory. Keep `safeguard_warmup=true` if using warmup.
7. The two JSON boxes are optional expert overrides. Examples: AdamW `{"betas":[0.9,0.96],"weight_decay":0.01}`; Prodigy `{"d_coef":0.5,"slice_p":11,"safeguard_warmup":true}`; MultiStep `{"milestones":[400,800,1200],"gamma":0.5}`; warm restarts `{"T_0":300,"T_mult":2,"eta_min":1e-7}`. Auto-tune keeps AdamW as the validated default; select Prodigy explicitly for an A/B run.
7. Keep **Fast in-memory checkpoints** off when you need crash-safe resume. RAM mode keeps only the best model weights in system memory and writes the final model once; VRAM mode is faster but consumes an additional model-sized GPU allocation. Both modes lose the current run's resume point if the process stops unexpectedly.
8. Use **Resume Checkpoint** only to continue a run. **Save Project** stores the visible training choices; **Start Training** launches the worker and writes a validated `ready` model when complete.
9. **Open TensorBoard** starts a local server for only the active/latest run and opens it automatically in the browser; it never points at the parent folder containing older runs.
10. The resulting model appears in Inference after pressing its model refresh button or reopening the GUI.""")
            with gr.Row(elem_classes="project-strip"):
                project = gr.Dropdown([NONE, *list_projects("training")], value=NONE, label="Training Project", allow_custom_value=True, scale=3)
                project_new = gr.Textbox(label="New Project Name", scale=2); project_create = gr.Button("Create Project"); project_save = gr.Button("💾 Save Project"); project_clone = gr.Button("Clone Project", variant="secondary"); project_refresh = gr.Button("↻", elem_classes="compact"); project_delete = gr.Button("🗑️", variant="stop", elem_classes="compact")
            training_project_status = gr.Markdown(elem_classes="status-card")
            with gr.Row():
                train_dataset = gr.Dropdown(B.list_datasets(), label="Dataset", info="Prepared dataset selected for this training project.", scale=4); dsref = gr.Button("↻", elem_classes="compact"); base = gr.Dropdown(B.BASE_VERSIONS, value="v2.0.3", label="Base Version", info="XTTS base checkpoint used before automatic pronunciation preparation.", scale=2)
            with gr.Row():
                epochs = gr.Slider(1, 100, 10, 1, label="Epochs", info="Complete passes over the dataset. Auto-tune now uses a longer convergence budget; holdout loss is the stopping signal.")
                batch = gr.Slider(1, 32, 2, 1, label="Batch Size", info="Per-step samples; constrained by available VRAM.")
                grad = gr.Slider(1, 64, 4, 1, label="Gradient Accumulation", info="Virtual batch multiplier when VRAM cannot hold the full batch.")
                lr = gr.Number(5e-6, label="Learning Rate", info="Optimizer step size; small datasets use a conservative XTTS fine-tuning rate.")
                maxlen = gr.Slider(4, 20, 11.6, .2, label="Max Audio Length (s)", info="Training audio ceiling; initialized from the dataset's language-aware policy.")
            with gr.Accordion("🧠 Optimizer, Scheduler & Warmup", open=False):
                with gr.Row():
                    optimizer = gr.Dropdown(B.TRAINING_OPTIMIZERS, value="AdamW", label="Optimizer", info="AdamW is the validated XTTS/AllTalk baseline. Prodigy is an experimental adaptive optimizer: use lr=1.0 and normally no external scheduler.")
                    scheduler = gr.Dropdown(B.TRAINING_SCHEDULERS, value="MultiStepLR", label="Learning-Rate Scheduler", info="Prodigy normally uses None (optimizer-managed). Other optimizers use the selected schedule per update or per epoch.")
                    warmup_steps = gr.Number(0, precision=0, minimum=0, maximum=1000000, label="Warmup Steps", info="Linear ramp before the scheduler. With per-epoch cadence this value means warmup epochs.")
                    scheduler_cadence = gr.Dropdown(B.TRAINING_SCHEDULER_CADENCES, value="Per optimizer step", label="Scheduler Cadence", info="Per optimizer step is recommended and respects gradient accumulation; per epoch is useful for slow schedules.")
                with gr.Row():
                    optimizer_params = gr.Textbox(value='{"betas":[0.9,0.96],"eps":1e-8,"weight_decay":0.01}', label="Optimizer Parameters JSON (optional)", lines=2, info="Merged with safe defaults. Prodigy example: {\"d_coef\":0.5,\"slice_p\":11,\"safeguard_warmup\":true}")
                    scheduler_params = gr.Textbox(value="{}", label="Scheduler Parameters JSON (optional)", lines=2, info="Overrides schedule defaults. Example: {\"T_0\":300,\"T_mult\":2,\"eta_min\":1e-7}")
            optimizer.change(
                optimizer_selection_ui,
                [optimizer, lr, optimizer_params, scheduler, warmup_steps, scheduler_params, scheduler_cadence],
                [lr, optimizer_params, scheduler, warmup_steps, scheduler_params, scheduler_cadence],
                queue=False,
            )
            with gr.Row():
                save_step = gr.Slider(50, 5000, 500, 50, label="Save Every Steps", info="Checkpoint interval; Auto-tune relates it to the training set size.")
                eval_steps = gr.Slider(50, 5000, 500, 50, label="Evaluate Every (Steps)", info="Cadence for quantitative validation on metadata_eval.csv. Auto-tune chooses a dataset-aware value.")
                training_seed = gr.Number(1234, precision=0, minimum=1, maximum=(1 << 31), label="Training Seed", info="Passed to Coqui XTTS training for reproducible data ordering and initialization.")
            with gr.Row():
                checkpoint_memory = gr.Checkbox(False, label="Fast in-memory checkpoints", info="RAM/VRAM mode avoids intermediate disk writes. Keep disabled for crash-safe recovery checkpoints.")
                checkpoint_target = gr.Dropdown(CHECKPOINT_TARGET_CHOICES, value="RAM (recommended)", label="Memory Target", info="RAM keeps VRAM available for training; VRAM is faster but reserves another model-sized GPU copy.", visible=False)
            checkpoint_memory.change(checkpoint_target_visibility, checkpoint_memory, checkpoint_target, queue=False)
            eval_enabled = gr.Checkbox(False, label="Enable Evaluation Zone", info="Keep off to skip prompted evaluation audio. XTTS holdout loss evaluation remains active.")
            with gr.Accordion("🎧 Prompted Holdout Evaluation", open=False, visible=False) as eval_zone:
                gr.Markdown("At each epoch boundary, XTTS synthesizes this same sentence using a deterministic reference selected from the validation holdout. The WAV is logged to TensorBoard alongside the quantitative holdout metrics. This is XTTS's native cadence; it is not a per-step Fish callback.", elem_classes="compact-status")
                eval_text = gr.Textbox(label="Fixed Eval Text", lines=3, value="This is a fixed evaluation sample generated during training.", info="Keep this sentence unchanged during a run so checkpoints are directly comparable.")
                gr.Markdown("Reference audio is selected automatically and deterministically from metadata_eval.csv. Enable the zone only when you want prompted comparison audio in TensorBoard.", elem_classes="compact-status")
            eval_enabled.change(lambda enabled: gr.update(visible=bool(enabled)), eval_enabled, eval_zone, queue=False)
            with gr.Row(elem_classes="workflow-actions"):
                autotune = gr.Button("⚙ Auto-tune Training", variant="secondary"); autotune_status = gr.Markdown(elem_classes="status-card")
            with gr.Row():
                resume = gr.Dropdown(["Fresh / None"], value="Fresh / None", label="Resume Checkpoint", info="Optional prior checkpoint from this training project."); rref = gr.Button("↻", elem_classes="compact")
            with gr.Row(elem_classes="workflow-actions"):
                start = gr.Button("🚀 Start Training", variant="primary", elem_id="start-training-btn"); stop = gr.Button("🛑 Stop Training", variant="stop", interactive=False, elem_id="stop-training-btn"); tb = gr.Button("📊 Open TensorBoard", variant="secondary")
            tr_progress = gr.HTML(value=training_progress_html())
            timer = gr.Timer(1)
            timer.tick(training_poll_ui, outputs=[tr_progress, start, stop], queue=False, show_progress="hidden")
            dsref.click(lambda: gr.update(choices=B.list_datasets()), outputs=train_dataset, queue=False)
            rref.click(checkpoint_update, [project, resume], resume, queue=False)
            project.change(load_training_project, [project, resume], [train_dataset, base, epochs, batch, grad, lr, optimizer, optimizer_params, scheduler, warmup_steps, scheduler_cadence, scheduler_params, maxlen, save_step, training_seed, checkpoint_memory, checkpoint_target, eval_steps, eval_enabled, eval_text, eval_zone, resume], queue=False)
            dataset_build_event.then(sync_dataset_completion, [dataset_project, project, train_dataset], [train_dataset, project], queue=False)
            autotune.click(autotune_ui, [train_dataset, base], [epochs, batch, grad, lr, optimizer, optimizer_params, scheduler, warmup_steps, scheduler_cadence, scheduler_params, maxlen, save_step, eval_steps, autotune_status], queue=False)
            start.click(
                training_controls_running,
                outputs=[start, stop],
                queue=False,
                show_progress="hidden",
                js="""() => {
                    const startRoot = document.getElementById('start-training-btn');
                    const stopRoot = document.getElementById('stop-training-btn');
                    const startButton = startRoot && startRoot.querySelector('button');
                    const stopButton = stopRoot && stopRoot.querySelector('button');
                    if (startButton) {
                        startButton.disabled = true;
                        startButton.textContent = 'Preparing...';
                        startButton.setAttribute('aria-busy', 'true');
                    }
                    if (stopButton) {
                        stopButton.disabled = true;
                        stopButton.textContent = '🛑 Stop Training';
                    }
                    return [];
                }""",
            ).then(
                start_training_ui,
                [project, train_dataset, base, epochs, batch, grad, lr, optimizer, optimizer_params, scheduler, warmup_steps, scheduler_cadence, scheduler_params, maxlen, save_step, training_seed, checkpoint_memory, checkpoint_target, eval_steps, eval_enabled, eval_text, resume],
                hidden,
                queue=False,
            )
            stop.click(
                stop_training_ui,
                outputs=hidden,
                queue=False,
                show_progress="hidden",
                js="""() => {
                    const stopRoot = document.getElementById('stop-training-btn');
                    const stopButton = stopRoot && stopRoot.querySelector('button');
                    if (stopButton) {
                        stopButton.disabled = true;
                        stopButton.textContent = 'Stopping...';
                    }
                    return [];
                }""",
            ).then(training_controls_idle, outputs=[start, stop], queue=False)

            def launch_tb(project_name):
                global _TENSORBOARD_PROC, _TENSORBOARD_LOGDIR, _TENSORBOARD_URL
                if not project_name:
                    return "Enter a project name."
                logdir = B.tensorboard_logdir(project_name)
                if logdir is None:
                    return "No TensorBoard run is initialized for this project yet. Start Training and try again after the worker creates its run."
                import subprocess as sp
                import socket
                import sys
                import threading
                import time
                import urllib.request
                import webbrowser

                resolved_logdir = str(logdir.resolve())
                if _TENSORBOARD_PROC is not None and _TENSORBOARD_PROC.poll() is None and _TENSORBOARD_LOGDIR == resolved_logdir:
                    webbrowser.open_new(_TENSORBOARD_URL)
                    return f"TensorBoard reopened for run `{logdir.name}`: {_TENSORBOARD_URL}"
                if _TENSORBOARD_PROC is not None and _TENSORBOARD_PROC.poll() is None:
                    _TENSORBOARD_PROC.terminate()
                    try:
                        _TENSORBOARD_PROC.wait(timeout=3)
                    except sp.TimeoutExpired:
                        _TENSORBOARD_PROC.kill()
                with socket.socket() as probe:
                    probe.bind(("127.0.0.1", 0))
                    port = probe.getsockname()[1]
                url = f"http://127.0.0.1:{port}"
                _TENSORBOARD_LOGDIR = resolved_logdir
                _TENSORBOARD_URL = url
                _TENSORBOARD_PROC = sp.Popen(
                    [sys.executable, "-m", "tensorboard.main", "--logdir", resolved_logdir, "--host", "127.0.0.1", "--port", str(port)],
                    cwd=B.ROOT,
                    stdout=sp.DEVNULL,
                    stderr=sp.STDOUT,
                )

                def open_when_ready(process, address):
                    deadline = time.time() + 30
                    while time.time() < deadline:
                        if process.poll() is not None:
                            break
                        try:
                            with urllib.request.urlopen(address, timeout=1):
                                webbrowser.open_new(address)
                                return
                        except Exception:
                            time.sleep(.25)
                    webbrowser.open_new(address)

                threading.Thread(target=open_when_ready, args=(_TENSORBOARD_PROC, url), name="tensorboard-browser", daemon=True).start()
                log(f"[UI] TensorBoard launched for run={logdir.name} · logdir={resolved_logdir}")
                return f"TensorBoard opening for run `{logdir.name}`: {url}"

            tb.click(launch_tb, project, hidden, queue=False)
            project_create.click(lambda name: create_project_ui(name, "training"), project_new, [project, training_project_status], queue=False)
            project_save.click(save_training_project_ui, [project, train_dataset, base, epochs, batch, grad, lr, optimizer, optimizer_params, scheduler, warmup_steps, scheduler_cadence, scheduler_params, maxlen, save_step, training_seed, checkpoint_memory, checkpoint_target, eval_steps, eval_enabled, eval_text, resume], [project, training_project_status], queue=False)
            project_clone.click(lambda name: clone_project_ui(name, "training"), project, [project, training_project_status], queue=False)
            project_refresh.click(lambda: project_update("training"), outputs=project, queue=False)
            project_delete.click(lambda name: delete_project_ui(name, "training"), project, [project, training_project_status], queue=False)

    with gr.Accordion("🖥️ Live Console & Training Log", open=True, elem_classes="console-accordion"):
        console = gr.HTML(value=html_view("XTTS-v2 Easy GUI Live Console")); console_timer = gr.Timer(.5)
        console_timer.tick(lambda: html_view("XTTS-v2 Easy GUI Console"), outputs=console, queue=False, show_progress="hidden")


if __name__ == "__main__":
    app.queue().launch(inbrowser=True, server_name="127.0.0.1", show_error=True)
