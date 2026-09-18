from __future__ import annotations
import argparse, json, math, os, shutil, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'base_models'
TRAINING=ROOT/'training'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
import TTS.tts.models.xtts as xtts_module
from xtts_backend import _xtts_load_audio

# Coqui's XTTS dataset helper imports this function by name. Patch it before
# the dataset loader is imported so training and inference use the same local
# decoder instead of torchaudio.load's 2.9 migration path.
xtts_module.load_audio = _xtts_load_audio
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig
from huggingface_hub import hf_hub_download


def report(message):
    print(f'[XTTS] {message}', flush=True)


def _json_object(raw, label):
    try:
        value = json.loads(str(raw or '{}').strip() or '{}')
    except json.JSONDecodeError as exc:
        raise ValueError(f'{label} must be a valid JSON object: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be a JSON object.')
    return value


def _optimizer_defaults(name, overrides):
    defaults = {
        'AdamW': {'betas': [0.9, 0.96], 'eps': 1e-8, 'weight_decay': 1e-2},
        'Adam': {'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0},
        'RAdam': {'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0},
        'RMSprop': {'alpha': 0.99, 'eps': 1e-8, 'momentum': 0.0, 'weight_decay': 0.0},
        'SGD': {'momentum': 0.9, 'weight_decay': 0.0, 'nesterov': False},
        # Prodigy uses lr=1 as a documented scale input and adapts its
        # effective step size through D-adaptation.  slice_p=11 keeps its
        # extra distance-estimation state bounded for the large XTTS GPT.
        'Prodigy': {
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'weight_decay': 0.0,
            'decouple': True,
            'use_bias_correction': False,
            'safeguard_warmup': True,
            'd0': 1e-6,
            'd_coef': 0.5,
            'slice_p': 11,
        },
    }
    if name not in defaults:
        raise ValueError(f'Unsupported optimizer: {name}')
    params = dict(defaults[name])
    params.update(overrides or {})
    return params


def _scheduler_defaults(name, params, total_updates, learning_rate):
    """Fill only safe defaults; explicit JSON values always win."""
    total_updates = max(1, int(total_updates))
    values = dict(params or {})
    if name in {None, '', 'None', 'None (optimizer-managed)'}:
        return {}
    defaults = {
        'MultiStepLR': {
            'milestones': [max(1, int(total_updates * 0.55)), max(1, int(total_updates * 0.78)), max(1, int(total_updates * 0.92))],
            'gamma': 0.5,
            'last_epoch': -1,
        },
        'CosineAnnealingWarmRestarts': {
            'T_0': max(25, min(total_updates, total_updates // 3 or 1)),
            'T_mult': 2,
            'eta_min': max(float(learning_rate) * 0.1, 1e-7),
            'last_epoch': -1,
        },
        'CosineAnnealingLR': {'T_max': total_updates, 'eta_min': max(float(learning_rate) * 0.1, 1e-7), 'last_epoch': -1},
        'ExponentialLR': {'gamma': 0.995, 'last_epoch': -1},
        'StepLR': {'step_size': max(1, total_updates // 3), 'gamma': 0.5, 'last_epoch': -1},
        'ConstantLR': {'factor': 1.0, 'total_iters': total_updates, 'last_epoch': -1},
        'LinearLR': {'start_factor': 1.0, 'end_factor': 0.1, 'total_iters': total_updates, 'last_epoch': -1},
        'PolynomialLR': {'total_iters': total_updates, 'power': 1.0, 'last_epoch': -1},
        'OneCycleLR': {
            'max_lr': float(learning_rate) * 10.0,
            'total_steps': total_updates,
            'pct_start': 0.1,
            'div_factor': 25.0,
            'final_div_factor': 10000.0,
            'cycle_momentum': False,
            'last_epoch': -1,
        },
        'CyclicLR': {
            'base_lr': max(float(learning_rate) * 0.1, 1e-7),
            'max_lr': float(learning_rate),
            'step_size_up': max(1, total_updates // 10),
            'cycle_momentum': False,
            'last_epoch': -1,
        },
    }
    if name not in defaults:
        raise ValueError(f'Unsupported scheduler: {name}')
    merged = dict(defaults[name])
    merged.update(values)
    return merged


def build_training_scheduler(optimizer, name, params, warmup_steps, total_updates, learning_rate):
    """Construct a native PyTorch schedule, optionally preceded by warmup.

    SequentialLR is stateful, so Coqui's existing checkpoint restore continues
    both the warmup phase and the selected decay schedule without extra files.
    """
    import torch

    if name in {None, '', 'None', 'None (optimizer-managed)'}:
        return None
    values = _scheduler_defaults(name, params, total_updates, learning_rate)
    warmup_start_factor = float(values.pop('warmup_start_factor', max(0.01, 1.0 / max(1, int(warmup_steps)))))
    scheduler_class = getattr(torch.optim.lr_scheduler, name)
    base_scheduler = scheduler_class(optimizer, **values)
    if int(warmup_steps) <= 0:
        return base_scheduler
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=max(0.0001, min(1.0, warmup_start_factor)),
        end_factor=1.0,
        total_iters=max(1, int(warmup_steps)),
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, base_scheduler],
        milestones=[max(1, int(warmup_steps))],
    )


class TunedGPTTrainer(GPTTrainer):
    """XTTS trainer that adds scheduler/warmup policy without forking Coqui."""

    def __init__(self, config, scheduler_name, scheduler_params, warmup_steps, total_updates):
        self._scheduler_name = scheduler_name
        self._scheduler_params = scheduler_params
        self._warmup_steps = int(warmup_steps)
        self._total_updates = int(total_updates)
        super().__init__(config)

    def get_optimizer(self):
        """Create Prodigy while retaining Coqui's XTTS parameter grouping.

        Coqui's stock helper only resolves names from torch.optim (plus RAdam),
        so a Prodigy run needs an explicit injection point. The two XTTS groups
        are kept intact: weights receive the configured decay and bias,
        normalization and embedding parameters remain decay-free. Prodigy
        requires the same lr in every non-zero group, which this configuration
        already guarantees.
        """
        if str(self.config.optimizer).strip() != 'Prodigy':
            return super().get_optimizer()
        try:
            from prodigyopt import Prodigy
        except Exception as exc:
            raise RuntimeError(
                'Prodigy requires prodigyopt in the project environment. Run install.bat and retry.'
            ) from exc

        import torch.nn as nn

        net = self.xtts.gpt
        if not self.config.optimizer_wd_only_on_weights:
            return Prodigy(
                self.xtts.gpt.parameters(),
                lr=float(self.config.lr),
                **self.config.optimizer_params,
            )

        norm_modules = (
            nn.BatchNorm2d,
            nn.InstanceNorm2d,
            nn.BatchNorm1d,
            nn.InstanceNorm1d,
            nn.BatchNorm3d,
            nn.InstanceNorm3d,
            nn.GroupNorm,
            nn.LayerNorm,
        )
        emb_modules = (nn.Embedding, nn.EmbeddingBag)
        param_names_notweights = set()
        all_param_names = set()
        param_map = {}
        for module_name, module in net.named_modules():
            for parameter_name, parameter in module.named_parameters():
                parameter.is_bias = parameter_name.endswith('.bias')
                parameter.is_weight = parameter_name.endswith('.weight')
                parameter.is_norm = isinstance(module, norm_modules)
                parameter.is_emb = isinstance(module, emb_modules)
                full_name = f'{module_name}.{parameter_name}' if module_name else parameter_name
                all_param_names.add(full_name)
                param_map[full_name] = parameter
                if parameter.is_bias or parameter.is_norm or parameter.is_emb:
                    param_names_notweights.add(full_name)

        params_names_notweights = sorted(param_names_notweights)
        params_names_weights = sorted(all_param_names ^ param_names_notweights)
        weight_decay = float(self.config.optimizer_params.get('weight_decay', 0.0) or 0.0)
        groups = [
            {'params': [param_map[name] for name in params_names_weights], 'weight_decay': weight_decay},
            {'params': [param_map[name] for name in params_names_notweights], 'weight_decay': 0.0},
        ]
        optimizer_params = dict(self.config.optimizer_params)
        optimizer_params.pop('weight_decay', None)
        optimizer_params.pop('lr', None)
        optimizer = Prodigy(groups, lr=float(self.config.lr), weight_decay=weight_decay, **optimizer_params)
        optimizer._group_names = [params_names_weights, params_names_notweights]
        return optimizer

    def get_scheduler(self, optimizer):
        return build_training_scheduler(
            optimizer,
            self._scheduler_name,
            self._scheduler_params,
            self._warmup_steps,
            self._total_updates,
            self.config.lr,
        )


class CheckpointStore:
    """Keep only the state needed for recovery and final model export.

    Coqui's default save_best_model writes a numbered full trainer checkpoint
    and then copies it to best_model.pth. That is particularly expensive for
    XTTS because the optimizer state is several gigabytes. This store keeps a
    single rolling recovery checkpoint on disk and a model-only best snapshot.
    In RAM/VRAM mode even those intermediate writes are skipped.
    """

    def __init__(self, output_path: Path, mode: str):
        self.output_path = Path(output_path)
        self.mode = mode if mode in {'disk', 'ram', 'vram'} else 'disk'
        self.best_path = self.output_path / 'best_model.pth'
        self.recovery_path = self.output_path / 'checkpoint.pth'
        self.best_state = None

    @staticmethod
    def _is_better(current, previous):
        if isinstance(current, dict) and isinstance(previous, dict):
            if current.get('eval_loss') is not None and previous.get('eval_loss') is not None:
                return current['eval_loss'] < previous['eval_loss']
            return current.get('train_loss', float('inf')) < previous.get('train_loss', float('inf'))
        return float(current) < float(previous)

    @staticmethod
    def _atomic_save(target: Path, writer):
        temporary = target.with_name(target.name + '.tmp')
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            writer(temporary)
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _save(self, target, config, model, current_step, epoch, optimizer=None, scheduler=None, scaler=None, model_loss=None):
        from trainer.io import save_model

        target = Path(target)

        def writer(path):
            save_model(
                config,
                model,
                path,
                current_step=current_step,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model_loss=model_loss,
            )

        self._atomic_save(target, writer)

    def _cache_in_memory(self, model):
        import torch

        requested = torch.device('cuda') if self.mode == 'vram' and torch.cuda.is_available() else torch.device('cpu')
        if self.mode == 'vram' and requested.type != 'cuda':
            report('VRAM checkpoint cache requested without CUDA; falling back to RAM.')
            self.mode = 'ram'
        try:
            with torch.no_grad():
                return {name: value.detach().to(device=requested, copy=True) for name, value in model.state_dict().items()}
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if requested.type != 'cuda':
                raise
            report(f'VRAM checkpoint cache unavailable ({exc}); falling back to RAM.')
            torch.cuda.empty_cache()
            self.mode = 'ram'
            with torch.no_grad():
                return {name: value.detach().to(device='cpu', copy=True) for name, value in model.state_dict().items()}

    def save_best(self, current_loss, previous_loss, config, model, current_step, epoch):
        if current_step <= 0 or not self._is_better(current_loss, previous_loss):
            return previous_loss
        if self.mode in {'ram', 'vram'}:
            self.best_state = self._cache_in_memory(model)
            report(f'Best model cached in {"VRAM" if self.mode == "vram" else "RAM"} · step={current_step} · no disk checkpoint written.')
        else:
            # Inference needs only model weights. The optimizer/scheduler/scaler
            # remain exclusively in the rolling recovery checkpoint below.
            self._save(self.best_path, config, model, current_step, epoch, model_loss=current_loss)
            report(f'Best model updated · step={current_step} · model-only checkpoint={self.best_path.name}')
        return current_loss

    def save_recovery(self, config, model, current_step, epoch, optimizer, scheduler, scaler, model_loss):
        if self.mode != 'disk':
            return
        # One stable filename avoids checkpoint_N plus retention copies.
        self._save(
            self.recovery_path,
            config,
            model,
            current_step,
            epoch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            model_loss=model_loss,
        )
        report(f'Recovery checkpoint refreshed · step={current_step} · optimizer state retained on disk.')

    def _restore_memory_best(self, model):
        if not self.best_state:
            return False
        current = model.state_dict()
        restored = {
            name: value.to(device=current[name].device, dtype=current[name].dtype, copy=True)
            for name, value in self.best_state.items()
        }
        model.load_state_dict(restored, strict=True)
        return True

    def export_model(self, target, config, model, current_step, epoch):
        target = Path(target)
        if self.mode == 'disk' and self.best_path.is_file():
            try:
                os.link(self.best_path, target)
                return
            except OSError:
                shutil.copy2(self.best_path, target)
                return
        if self.mode in {'ram', 'vram'} and self._restore_memory_best(model):
            report(f'Best in-memory weights restored for final export from {self.mode.upper()} cache.')
        self._save(target, config, model, current_step, epoch)


class EfficientTrainer(Trainer):
    """Trainer variant with XTTS-sized checkpoint I/O kept under control."""

    def __init__(self, *args, checkpoint_mode='disk', **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_store = CheckpointStore(self.output_path, checkpoint_mode)

    def save_best_model(self):
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train) or float('inf')
        current_loss = {'train_loss': train_loss, 'eval_loss': eval_loss}
        self.best_loss = self.checkpoint_store.save_best(
            current_loss,
            self.best_loss,
            self.config,
            self._get_model(),
            self.total_steps_done,
            self.epochs_done,
        )

    def save_checkpoint(self):
        eval_loss = self._pick_target_avg_loss(self.keep_avg_eval)
        train_loss = self._pick_target_avg_loss(self.keep_avg_train)
        self.checkpoint_store.save_recovery(
            self.config,
            self._get_model(),
            self.total_steps_done,
            self.epochs_done,
            self.optimizer,
            self.scheduler,
            self.scaler if self.use_amp_scaler else None,
            {'train_loss': train_loss, 'eval_loss': eval_loss},
        )


def ensure_base(version):
    out=BASE/version;out.mkdir(parents=True,exist_ok=True)
    rev=None if version=='main' else version
    for fn in ('config.json','model.pth','vocab.json','speakers_xtts.pth'):
        if not (out/fn).exists():hf_hub_download('coqui/XTTS-v2',fn,revision=rev,local_dir=out)
    for fn in ('dvae.pth','mel_stats.pth'):
        if not (out/fn).exists():hf_hub_download('coqui/XTTS-v2',fn,revision='main',local_dir=out)
    return out


def validate_tokenizer_checkpoint(vocab: Path, checkpoint: Path) -> int:
    from tokenizers import Tokenizer
    import torch
    token_count = Tokenizer.from_file(str(vocab)).get_vocab_size(with_added_tokens=True)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    emb = state.get("gpt.text_embedding.weight")
    head = state.get("gpt.text_head.weight")
    if emb is None or head is None:
        raise RuntimeError("Checkpoint does not contain XTTS GPT text embedding/head weights.")
    if int(emb.shape[0]) != token_count or int(head.shape[0]) != token_count:
        raise RuntimeError(f"Tokenizer/checkpoint mismatch: tokenizer={token_count}, embedding={emb.shape[0]}, head={head.shape[0]}")
    return token_count

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',required=True);ap.add_argument('--dataset',required=True);ap.add_argument('--base-version',default='v2.0.3')
    ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--batch-size',type=int,default=2);ap.add_argument('--grad-accum',type=int,default=4);ap.add_argument('--lr',type=float,default=5e-6);ap.add_argument('--max-audio-length',type=float,default=11.6);ap.add_argument('--save-step',type=int,default=500);ap.add_argument('--resume',default='');ap.add_argument('--vocab',default='');ap.add_argument('--base-checkpoint',default='');ap.add_argument('--checkpoint-mode',choices=('disk','ram','vram'),default='disk')
    ap.add_argument('--optimizer',choices=('AdamW','Adam','RAdam','RMSprop','SGD','Prodigy'),default='AdamW'); ap.add_argument('--optimizer-params',default='{}')
    ap.add_argument('--scheduler',choices=('None (optimizer-managed)','MultiStepLR','CosineAnnealingWarmRestarts','CosineAnnealingLR','ExponentialLR','StepLR','ConstantLR','LinearLR','PolynomialLR','OneCycleLR','CyclicLR'),default='MultiStepLR'); ap.add_argument('--warmup-steps',type=int,default=0); ap.add_argument('--scheduler-cadence',choices=('Per optimizer step','Per epoch'),default='Per optimizer step'); ap.add_argument('--scheduler-params',default='{}')
    ap.add_argument('--progress-file',default=''); ap.add_argument('--training-seed',type=int,default=1234)
    ap.add_argument('--eval-steps',type=int,default=500); ap.add_argument('--eval-enabled',action='store_true'); ap.add_argument('--eval-text',default='')
    a=ap.parse_args(); dataset=Path(a.dataset).resolve(); out=TRAINING/a.project; run=out/'run';ready=out/'ready';run.mkdir(parents=True,exist_ok=True)
    optimizer_params=_optimizer_defaults(a.optimizer, _json_object(a.optimizer_params, 'Optimizer parameters'))
    scheduler_params=_json_object(a.scheduler_params, 'Scheduler parameters')
    if a.optimizer == 'Prodigy':
        try:
            a.lr = 1.0
        except (TypeError, ValueError):
            a.lr = 1.0
        if a.scheduler in {'None', 'None (optimizer-managed)'}:
            a.scheduler = 'None (optimizer-managed)'
        if int(a.warmup_steps) > 0 and not bool(optimizer_params.get('safeguard_warmup', True)):
            report('Prodigy warmup requested without safeguard_warmup; enabling safeguard_warmup for stable D-adaptation.')
            optimizer_params['safeguard_warmup'] = True
    progress_path=Path(a.progress_file).resolve() if a.progress_file else out/'progress.json'
    progress_path.parent.mkdir(parents=True,exist_ok=True)
    run_dir=''
    started=time.time()
    report(f'Worker started · project={a.project} · dataset={dataset.name} · base={a.base_version}')
    report(f'Training parameters · epochs={a.epochs} · batch={a.batch_size} · grad_accum={a.grad_accum} · lr={a.lr} · seed={a.training_seed}')
    report(f'Optimizer/scheduler · {a.optimizer} {json.dumps(optimizer_params, ensure_ascii=False, sort_keys=True)} · {a.scheduler} {json.dumps(scheduler_params, ensure_ascii=False, sort_keys=True)} · cadence={a.scheduler_cadence} · warmup={a.warmup_steps}')
    if a.optimizer == 'Prodigy':
        report('Prodigy mode · lr=1.0 is the optimizer scale input; progress will report the adaptive effective LR and D estimate.')
    report(f'Checkpoint policy · {a.checkpoint_mode} ({"one rolling recovery file" if a.checkpoint_mode == "disk" else "no intermediate checkpoint writes"})')
    report(f'Evaluation parameters · quantitative every {a.eval_steps} steps · prompted audio={bool(a.eval_enabled)}')

    def write_progress(payload):
        if run_dir:
            payload.setdefault('run_dir', run_dir)
        data=json.dumps(payload,ensure_ascii=False)
        temporary=progress_path.with_suffix(progress_path.suffix+'.tmp')
        try:
            temporary.write_text(data,encoding='utf-8')
            for _ in range(10):
                try:
                    os.replace(temporary,progress_path)
                    return
                except PermissionError:
                    # The GUI timer can briefly hold progress.json open on
                    # Windows. A progress refresh must never abort training.
                    time.sleep(0.02)
            try:
                with progress_path.open('w',encoding='utf-8') as handle:
                    handle.write(data)
                    handle.flush()
            except OSError:
                pass
        except OSError:
            pass
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    write_progress({'schema':1,'running':True,'project':a.project,'phase':'Preparing XTTS training','pct':0.0,'epoch':0,'total_epochs':a.epochs,'step':0,'total_steps':0,'loss':None,'learning_rate':float(a.lr),'base_learning_rate':float(a.lr),'adaptive_scale':None,'optimizer':a.optimizer,'elapsed':0.0,'eta':None})
    report('Loading XTTS base artifacts...')
    base=ensure_base(a.base_version); lang=(dataset/'lang.txt').read_text(encoding='utf-8').strip()
    report(f'Language resolved: {lang}')
    vocab=Path(a.vocab) if a.vocab else base/'vocab.json'; checkpoint=Path(a.base_checkpoint) if a.base_checkpoint else base/'model.pth'
    vocab=vocab.resolve(); checkpoint=checkpoint.resolve()
    report(f'Using tokenizer: {vocab}')
    report(f'Using checkpoint: {checkpoint}')
    token_count=validate_tokenizer_checkpoint(vocab, checkpoint)
    report(f'Tokenizer/checkpoint validated · vocabulary size={token_count}')
    cfg_dataset=BaseDatasetConfig(formatter='coqui',dataset_name='xtts_easy',path=str(dataset),meta_file_train=str(dataset/'metadata_train.csv'),meta_file_val=str(dataset/'metadata_eval.csv'),language=lang)
    model_args=GPTArgs(max_conditioning_length=132300,min_conditioning_length=66150,max_wav_length=int(a.max_audio_length*22050),max_text_length=240,mel_norm_file=str(base/'mel_stats.pth'),dvae_checkpoint=str(base/'dvae.pth'),xtts_checkpoint=str(checkpoint),tokenizer_file=str(vocab),gpt_number_text_tokens=token_count,gpt_num_audio_tokens=1026,gpt_start_audio_token=1024,gpt_stop_audio_token=1025,gpt_use_masking_gt_prompt_approach=True,gpt_use_perceiver_resampler=True)
    audio=XttsAudioConfig(sample_rate=22050,dvae_sample_rate=22050,output_sample_rate=24000)
    eval_split_max_size=256
    report('Loading train/eval samples from metadata...')
    train_samples,eval_samples=load_tts_samples([cfg_dataset],eval_split=True,eval_split_max_size=eval_split_max_size,eval_split_size=0.01)
    report(f'Dataset loaded · train samples={len(train_samples)} · eval samples={len(eval_samples)}')
    evaluation_text=str(a.eval_text or '').strip()
    evaluation_reference=None
    test_sentences=[]
    if a.eval_enabled:
        if not evaluation_text:
            evaluation_text='This is a fixed evaluation sample generated during training.'
        if not eval_samples:
            raise RuntimeError('Evaluation Zone is enabled, but metadata_eval.csv has no usable samples.')
        evaluation_reference=min(eval_samples,key=lambda sample: str(sample.get('audio_file','')).casefold())
        reference_audio=Path(str(evaluation_reference.get('audio_file',''))).resolve()
        if not reference_audio.is_file():
            raise RuntimeError(f'Evaluation reference audio not found: {reference_audio}')
        test_sentences=[{'text':evaluation_text,'speaker_wav':str(reference_audio),'language':str(evaluation_reference.get('language') or lang)}]
        report(f'Prompted evaluation reference selected deterministically: {reference_audio.name}')
    else:
        report('Prompted evaluation audio disabled; quantitative holdout evaluation remains active.')
    evaluation_record={'schema':1,'enabled':bool(a.eval_enabled),'text':evaluation_text,'every_steps':int(max(0,a.eval_steps)),'quantitative_holdout':'metadata_eval.csv','prompted_cadence':'epoch_boundary' if a.eval_enabled else None,'reference_audio':str(Path(str(evaluation_reference.get('audio_file',''))).resolve()) if evaluation_reference else None,'reference_language':str(evaluation_reference.get('language') or lang) if evaluation_reference else None}
    (out/'evaluation.json').write_text(json.dumps(evaluation_record,ensure_ascii=False,indent=2),encoding='utf-8')
    steps_per_epoch=max(1, math.ceil(len(train_samples) / max(1, int(a.batch_size))))
    total_updates=max(1, math.ceil(steps_per_epoch / max(1, int(a.grad_accum))) * max(1, int(a.epochs)))
    if a.scheduler in {'OneCycleLR', 'CyclicLR'} and a.scheduler_cadence == 'Per epoch':
        report(f'{a.scheduler} requires per-update stepping; overriding cadence to Per optimizer step.',)
        a.scheduler_cadence='Per optimizer step'
    scheduler_horizon=total_updates if a.scheduler_cadence == 'Per optimizer step' else max(1, int(a.epochs))
    scheduler_params=_scheduler_defaults(a.scheduler, scheduler_params, scheduler_horizon, a.lr)
    a.warmup_steps=min(max(0, int(a.warmup_steps)), scheduler_horizon)
    if a.optimizer == 'Prodigy' and a.scheduler not in {'None', 'None (optimizer-managed)'}:
        report(f'Prodigy is running with explicit scheduler {a.scheduler}; official guidance generally prefers no scheduler or constant decay.',)
    report(f'Scheduler horizon · {scheduler_horizon:,} {"optimizer updates" if a.scheduler_cadence == "Per optimizer step" else "epochs"} ({total_updates:,} optimizer updates in the full run).')
    config=GPTTrainerConfig(epochs=a.epochs,output_path=str(run),model_args=model_args,run_name='XTTS_Easy_GUI',project_name=a.project,dashboard_logger='tensorboard',audio=audio,batch_size=a.batch_size,batch_group_size=48,eval_batch_size=a.batch_size,num_loader_workers=0 if lang=='ja' else 4,eval_split_max_size=eval_split_max_size,run_eval=True,run_eval_steps=(int(a.eval_steps) if int(a.eval_steps)>0 else None),print_eval=True,print_step=20,plot_step=100,log_model_step=100,save_step=a.save_step,save_n_checkpoints=1,save_checkpoints=(a.checkpoint_mode == 'disk'),optimizer=a.optimizer,optimizer_wd_only_on_weights=True,optimizer_params=optimizer_params,lr=a.lr,lr_scheduler=None,lr_scheduler_params={},scheduler_after_epoch=(a.scheduler_cadence == 'Per epoch'),training_seed=a.training_seed,test_sentences=test_sentences)
    report('Initializing XTTS GPT trainer and loading model weights...')
    model=TunedGPTTrainer(config,a.scheduler,scheduler_params,a.warmup_steps,total_updates)
    report('XTTS GPT trainer initialized.')

    def as_float(value):
        try:
            if hasattr(value,'detach'): value=value.detach().cpu()
            return float(value)
        except (TypeError,ValueError):
            return None

    last_reported_step=-1
    def publish_progress(trainer, phase='Training', running=True):
        nonlocal last_reported_step
        if getattr(getattr(trainer,'args',None),'rank',None) not in (None,0): return
        step=int(getattr(trainer,'total_steps_done',0) or 0)
        loader=getattr(trainer,'train_loader',None)
        per_epoch=len(loader) if loader is not None else 0
        total_steps=int(per_epoch*int(config.epochs))
        elapsed=max(0.0,time.time()-started)
        pct=(step/total_steps*100.0) if total_steps else 0.0
        eta=(elapsed/step*(total_steps-step)) if running and step and total_steps>step else None
        loss=None
        averages=getattr(getattr(trainer,'keep_avg_train',None),'avg_values',{}) or {}
        if isinstance(averages,dict):
            loss=as_float(averages.get('loss'))
            if loss is None:
                for value in averages.values():
                    loss=as_float(value)
                    if loss is not None: break
        learning_rate=None
        base_learning_rate=None
        adaptive_scale=None
        optimizer_name=None
        optimizer=getattr(trainer, 'optimizer', None)
        if isinstance(optimizer, (list, tuple)):
            optimizer=optimizer[0] if optimizer else None
        if isinstance(optimizer, dict):
            optimizer=next(iter(optimizer.values()), None)
        try:
            if optimizer is not None and optimizer.param_groups:
                group=optimizer.param_groups[0]
                optimizer_name=optimizer.__class__.__name__
                base_learning_rate=float(group.get('lr'))
                if optimizer_name == 'Prodigy':
                    adaptive_scale=float(group.get('d', 0.0) or 0.0)
                    bias_correction=1.0
                    if bool(group.get('use_bias_correction', False)):
                        beta1, beta2=group.get('betas', (0.9, 0.999))
                        step_count=int(group.get('k', 0) or 0)
                        bias_correction=((1.0 - float(beta2) ** (step_count + 1)) ** 0.5) / max(1e-12, 1.0 - float(beta1) ** (step_count + 1))
                    learning_rate=adaptive_scale * base_learning_rate * bias_correction
                else:
                    learning_rate=base_learning_rate
        except (AttributeError, TypeError, ValueError):
            learning_rate=None
            base_learning_rate=None
            adaptive_scale=None
            optimizer_name=None
        report_every=max(1,int(config.print_step or 20))
        if step == 0 or step - last_reported_step >= report_every:
            loss_label='--' if loss is None else f'{loss:.5f}'
            lr_label='--' if learning_rate is None else f'{learning_rate:.3e}'
            adaptive_label='' if adaptive_scale is None else f' · d {adaptive_scale:.3e}'
            report(f'[UI-PROGRESS] {phase} · epoch {int(getattr(trainer,"epochs_done",0) or 0)+1}/{int(config.epochs)} · step {step}/{total_steps or "--"} · loss {loss_label} · effective lr {lr_label}{adaptive_label} · elapsed {elapsed:.0f}s')
            last_reported_step=step
        write_progress({'schema':1,'running':running,'project':a.project,'phase':phase,'pct':min(100.0,max(0.0,pct)),'epoch':int(getattr(trainer,'epochs_done',0) or 0)+1,'total_epochs':int(config.epochs),'step':step,'total_steps':total_steps,'loss':loss,'learning_rate':learning_rate,'base_learning_rate':base_learning_rate,'adaptive_scale':adaptive_scale,'optimizer':optimizer_name or a.optimizer,'elapsed':elapsed,'eta':eta})

    callbacks={'on_train_step_end':lambda trainer: publish_progress(trainer)}
    report('Creating Coqui Trainer and starting fit...')
    trainer=EfficientTrainer(TrainerArgs(restore_path=a.resume or None,skip_train_epoch=False,start_with_eval=False,grad_accum_steps=a.grad_accum),config,output_path=str(run),model=model,train_samples=train_samples,eval_samples=eval_samples,callbacks=callbacks,checkpoint_mode=a.checkpoint_mode)
    run_dir=str(Path(trainer.output_path).resolve())
    write_progress({'schema':1,'running':True,'project':a.project,'phase':'Starting training','pct':0.0,'epoch':0,'total_epochs':a.epochs,'step':0,'total_steps':0,'loss':None,'learning_rate':float(a.lr),'base_learning_rate':float(a.lr),'adaptive_scale':None,'optimizer':a.optimizer,'elapsed':max(0.0,time.time()-started),'eta':None,'run_dir':run_dir})
    report(f'Checkpoint store ready · mode={trainer.checkpoint_store.mode} · run={run_dir}')
    try:
        trainer.fit()
    except BaseException as exc:
        report(f'Training failed: {exc}')
        failed={'schema':1,'running':False,'project':a.project,'phase':f'Training failed: {exc}','pct':0.0,'epoch':0,'total_epochs':a.epochs,'step':0,'total_steps':0,'loss':None,'elapsed':max(0.0,time.time()-started),'eta':None}
        try: write_progress(failed)
        except OSError: pass
        raise
    try:
        final=json.loads(progress_path.read_text(encoding='utf-8')) if progress_path.is_file() else {}
    except (OSError,TypeError,ValueError,json.JSONDecodeError):
        final={}
    report('Trainer fit completed. Exporting the best checkpoint and runtime assets...')
    final.update({'running':False,'project':a.project,'phase':'Training complete','pct':100.0,'elapsed':max(0.0,time.time()-started),'eta':None})
    write_progress(final)
    staging=out/'ready.staging'; shutil.rmtree(staging,ignore_errors=True); staging.mkdir(parents=True,exist_ok=True)
    trainer.checkpoint_store.export_model(staging/'model.pth', config, trainer._get_model(), trainer.total_steps_done, trainer.epochs_done)
    shutil.copy2(vocab,staging/'vocab.json')
    source_config = checkpoint.parent/'config.json' if a.base_checkpoint and (checkpoint.parent/'config.json').exists() else base/'config.json'
    shutil.copy2(source_config,staging/'config.json');shutil.copy2(base/'speakers_xtts.pth',staging/'speakers_xtts.pth')
    for fn in ('dvae.pth','mel_stats.pth'):
        if (base/fn).exists(): shutil.copy2(base/fn,staging/fn)
    if (out/'evaluation.json').is_file():
        shutil.copy2(out/'evaluation.json',staging/'evaluation.json')
    dataset_lexicon=dataset/'pronunciation_lexicon.json'
    if dataset_lexicon.is_file():
        shutil.copy2(dataset_lexicon,staging/'pronunciation_lexicon.json')
    elif checkpoint.parent.joinpath('pronunciation_lexicon.json').is_file():
        shutil.copy2(checkpoint.parent/'pronunciation_lexicon.json',staging/'pronunciation_lexicon.json')
    ref=Path(train_samples[0]['audio_file']) if train_samples else None
    if ref and ref.exists():shutil.copy2(ref,staging/'reference.wav')
    manifest={'schema':3,'project':a.project,'language':lang,'base_version':a.base_version,'epochs':a.epochs,'batch_size':a.batch_size,'grad_accum':a.grad_accum,'learning_rate':a.lr,'learning_rate_semantics':'Prodigy adaptive scale input' if a.optimizer == 'Prodigy' else 'optimizer step size','training_seed':a.training_seed,'max_audio_length':a.max_audio_length,'optimizer':a.optimizer,'optimizer_params':optimizer_params,'scheduler':a.scheduler,'scheduler_params':scheduler_params,'warmup_steps':int(a.warmup_steps),'scheduler_cadence':a.scheduler_cadence,'scheduler_total_updates':total_updates,'scheduler_horizon':scheduler_horizon,'eval':evaluation_record,'expanded_tokenizer':bool(a.vocab),'pronunciation_pipeline':'automatic','pronunciation_lexicon':str(staging/'pronunciation_lexicon.json') if (staging/'pronunciation_lexicon.json').is_file() else 'automatic-empty','token_count':token_count,'checkpoint_mode':trainer.checkpoint_store.mode,'checkpoint_source':str(trainer.checkpoint_store.best_path if trainer.checkpoint_store.best_path.is_file() else staging/'model.pth')}
    (staging/'training_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    shutil.rmtree(ready,ignore_errors=True); staging.replace(ready)
    report(f'Ready model exported: {ready}')
if __name__=='__main__':main()
