"""Bounded single-GPU Melo GAN finetuning with explicit success/failure artifacts.

Uses official SynthesizerTrn, MPD, duration discriminator and all five generator
losses. This is a local bounded trainer, not a claim that official train.py ran.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time
import random

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from melo_training_core import model_from_config, setup
from run_korean_melo_spec import ROOT, save, sha256
from melo_checkpoint import save_checkpoint, verify_checkpoint, restore_rng


def check_dataset(cfg):
    licensed=cfg['experiment'].get('research_policy_validated') is True
    if not cfg['experiment'].get('permission_declarations_validated') and not licensed:
        raise ValueError('Input permission validation required')
    rows = json.loads(Path(cfg['experiment']['dataset_manifest']).read_text())['utterances']
    if not rows or len({r['utterance_id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate dataset')
    for row in rows:
        for name in ('wav', 'features'):
            if sha256(row[name]) != row[name+'_sha256']:
                raise ValueError('Dataset integrity check failed')
    # Revalidate original human declarations, never infer review from a config flag.
    from run_korean_melo_spec import check_manifest
    source = Path(cfg['experiment']['source_spec_run'])/'input_manifest.json'
    checked = check_manifest(source, allow_nonhuman=True,allow_licensed_research=licensed)
    ids = {r['utterance_id']: r for r in checked}
    for row in rows:
        if row['utterance_id'] not in ids or any(row[k] != ids[row['utterance_id']][k] for k in ('text','split','input_kind')):
            raise ValueError('Training dataset differs from validated source')
    train = [r for r in rows if r['split']=='train']
    if not train:
        raise ValueError('No training rows')
    return train


def finite_grad(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        raise ValueError('Missing parameter gradients')
    norm=torch.nn.utils.clip_grad_norm_(module.parameters(),float('inf'),error_if_nonfinite=True)
    if not torch.isfinite(norm):
        raise ValueError('Missing or nonfinite parameter gradients')
    if not norm>0:
        raise ValueError('Zero parameter gradients')


def train(config_path, assets, steps, *, resume=None, checkpoint_every=50, interrupt_after_checkpoint=None, output_root=None):
    if not 1 <= steps <= 10000:
        raise ValueError('steps must be in 1..10000')
    cfg = json.loads(Path(config_path).read_text())
    train_rows = check_dataset(cfg)
    if checkpoint_every<1:
        raise ValueError('checkpoint_every must be positive')
    resumed=verify_checkpoint(resume,cfg) if resume else None
    start_step=resumed['step'] if resumed else 0
    if start_step>=steps:
        raise ValueError('Target steps must exceed resumed step')
    setup(assets)
    from melo.models import MultiPeriodDiscriminator, DurationDiscriminator
    from melo import commons
    from melo.mel_processing import spectrogram_torch, spec_to_mel_torch, mel_spectrogram_torch
    from melo.losses import discriminator_loss, generator_loss, feature_loss, kl_loss
    torch.set_num_threads(4)
    torch.manual_seed(cfg['train']['seed'])
    np.random.seed(cfg['train']['seed'])
    random.seed(cfg['train']['seed'])
    torch.backends.cudnn.benchmark = False
    output_root=Path(output_root) if output_root else ROOT/'runs'
    output_root.mkdir(parents=True,exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='melo_train_'+cfg['experiment']['condition']+'_', dir=output_root))
    save(folder/'config.json', cfg)
    report = {'status': 'FAIL', 'completed_steps': start_step, 'requested_steps': steps,
        'scope': cfg['experiment']['scope'], 'condition': cfg['experiment']['condition'],
        'initial_checkpoint_sha256': sha256(cfg['experiment']['generator_initialization']),
        'torch': torch.__version__, 'precision': 'bf16', 'losses': [],
        'not_measured': ['defense efficacy', 'converged model quality'],
        'start_step':start_step,'resume_from':str(resume) if resume else None,'checkpoints':[],
        'trainer_sha256':sha256(Path(__file__))}
    started = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        initial_path=Path(resume)/resumed['files']['generator']['path'] if resumed else cfg['experiment']['generator_initialization']
        generator = model_from_config(cfg, initial_path).train()
        discriminator = MultiPeriodDiscriminator(cfg['model']['use_spectral_norm']).cuda().train()
        duration = DurationDiscriminator(cfg['model']['hidden_channels'],cfg['model']['hidden_channels'],
            3,.1,gin_channels=cfg['model']['gin_channels']).cuda().train()
        nets = (generator, discriminator, duration)
        optimizers = [torch.optim.AdamW(net.parameters(),lr=cfg['train']['learning_rate'],
            betas=tuple(cfg['train']['betas']),eps=cfg['train']['eps']) for net in nets]
        og, od, ou = optimizers
        initial_probe = generator.dec.conv_post.weight.detach().clone()
        if resumed:
            state=torch.load(Path(resume)/resumed['files']['state']['path'],map_location='cpu',weights_only=True)
            if state['step']!=start_step:
                raise ValueError('Checkpoint step mismatch')
            discriminator.load_state_dict(state['discriminator'],strict=True)
            duration.load_state_dict(state['duration_discriminator'],strict=True)
            for optimizer,saved in zip(optimizers,state['optimizers']):
                optimizer.load_state_dict(saved)
            report['restored_optimizer_steps']=[sorted({int(v['step']) for v in o.state.values() if 'step' in v}) for o in optimizers]
            report['losses']=state['history']
            initial_probe=state['initial_probe'].cuda()
            restore_rng(state['rng'])
            del state
            report['resume_integrity_verified']=True
        d = cfg['data']; seg = cfg['train']['segment_size']; frames=seg//d['hop_length']
        for step in range(start_step,steps):
            torch.cuda.synchronize()
            step_started=time.perf_counter()
            generator.current_mas_noise_scale = max(
                generator.mas_noise_scale_initial-generator.noise_scale_delta*step, 0.)
            row = train_rows[step%len(train_rows)]
            feature = torch.load(row['features'],map_location='cpu',weights_only=True)
            audio, rate = sf.read(row['wav'],dtype='float32')
            if rate!=d['sampling_rate'] or audio.ndim!=1 or not np.isfinite(audio).all():
                raise ValueError('Invalid training WAV')
            y=torch.from_numpy(audio)[None,None].cuda()
            spec=spectrogram_torch(y[:,0],d['filter_length'],rate,d['hop_length'],d['win_length'],center=False)
            x=feature['phone_ids'][None].cuda(); n=x.shape[-1]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=generator(x,torch.tensor([n],device='cuda'),spec,
                    torch.tensor([spec.shape[-1]],device='cuda'),torch.tensor([0],device='cuda'),
                    feature['tone_ids'][None].cuda(),feature['language_ids'][None].cuda(),
                    torch.zeros(1,1024,n,device='cuda'),feature['ja_bert'][None].cuda())
                yhat, length, attn, slices, xmask, zmask, latent, hidden = out
            # FFT/mel/KL in float32, matching the official mixed-precision loss boundary.
            mel=spec_to_mel_torch(spec.float(),d['filter_length'],d['n_mel_channels'],rate,d['mel_fmin'],d['mel_fmax'])
            target=commons.slice_segments(mel,slices,frames)
            predicted=mel_spectrogram_torch(yhat[:,0].float(),d['filter_length'],d['n_mel_channels'],
                rate,d['hop_length'],d['win_length'],d['mel_fmin'],d['mel_fmax'])
            yseg=commons.slice_segments(y,slices*d['hop_length'],seg)
            for net in (discriminator,duration):
                net.requires_grad_(True)
            od.zero_grad(set_to_none=True); ou.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                real,fake,_,_=discriminator(yseg,yhat.detach())
                dr,dg=duration(hidden[0].detach(),xmask.detach(),hidden[1].detach(),hidden[2].detach())
            dl=discriminator_loss(real,fake)[0]; ul=discriminator_loss(dr,dg)[0]
            dl.backward(); finite_grad(discriminator); od.step(); od.zero_grad(set_to_none=True)
            ul.backward(); finite_grad(duration); ou.step(); ou.zero_grad(set_to_none=True)
            for net in (discriminator,duration):
                net.requires_grad_(False)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                real,fake,fr,fg=discriminator(yseg,yhat)
                dr,dg=duration(hidden[0],xmask,hidden[1],hidden[2])
            z,zp,mp,lp,mq,lq=latent
            parts={'mel':F.l1_loss(target,predicted)*cfg['train']['c_mel'],
                'kl':kl_loss(zp,lq,mp,lp,zmask)*cfg['train']['c_kl'],
                'duration':length.float().sum(), 'feature':feature_loss(fr,fg),
                'adversarial':generator_loss(fake)[0], 'duration_adversarial':generator_loss(dg)[0]}
            total=sum(parts.values())
            if not all(torch.isfinite(v).all() for v in [dl,ul,total,*parts.values()]):
                raise ValueError('Nonfinite training loss')
            og.zero_grad(set_to_none=True); total.backward(); finite_grad(generator)
            og.step(); og.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step_seconds=time.perf_counter()-step_started
            values={'step':step+1,'utterance_id':row['utterance_id'],'generator':float(total.detach()),
                'discriminator':float(dl.detach()),'duration_discriminator':float(ul.detach()),
                'target_rms':float(yseg.square().mean().sqrt()),
                'prediction_rms':float(yhat.detach().float().square().mean().sqrt()),
                'step_seconds_excluding_checkpoint':step_seconds,
                **{k:float(v.detach()) for k,v in parts.items()}}
            report['losses'].append(values); report['completed_steps']=step+1
            print(json.dumps(values),flush=True)
            if (step+1)%checkpoint_every==0 or step+1==steps:
                directory=folder/'checkpoints'/f'step_{step+1:06d}'
                committed=save_checkpoint(directory,cfg,nets,optimizers,step+1,report['losses'],initial_probe)
                report['checkpoints'].append({'directory':str(directory),**committed})
                report['checkpoint_directory']=str(directory)
                report['checkpoint']=str(directory/committed['files']['generator']['path'])
                report['checkpoint_sha256']=committed['files']['generator']['sha256']
                print(json.dumps({'checkpoint_step':step+1,'save_seconds':committed['save_seconds']}),flush=True)
                if interrupt_after_checkpoint==step+1:
                    report['status']='INTERRUPTED_TEST_AFTER_COMMITTED_CHECKPOINT'
                    raise KeyboardInterrupt('Controlled checkpoint/recovery test')
        torch.cuda.synchronize()
        delta=float((generator.dec.conv_post.weight.detach()-initial_probe).abs().max())
        if not delta>0:
            raise ValueError('Optimizer did not change generator weights')
        report.update(status='TRAINING_SMOKE_PASS',parameter_probe_max_delta=delta,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['seconds']=time.perf_counter()-started
        if torch.cuda.is_available():
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        report['session_completed_steps']=report['completed_steps']-start_step
        current=[r['step_seconds_excluding_checkpoint'] for r in report['losses'] if r['step']>start_step]
        if current:
            report['step_timing']={'mean_seconds':float(np.mean(current)),
                'median_seconds':float(np.median(current)), 'p90_seconds':float(np.quantile(current,.9)),
                'total_step_seconds':float(sum(current)),'count':len(current)}
        save(folder/'training_report.json',report)
        print(json.dumps({'run':str(folder),'status':report['status']}),flush=True)
    return folder


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1)
    p.add_argument('--resume',type=Path)
    p.add_argument('--checkpoint-every',type=int,default=50)
    p.add_argument('--interrupt-after-checkpoint',type=int)
    p.add_argument('--output-root',type=Path)
    args=p.parse_args()
    train(args.config,args.assets,args.steps,resume=args.resume,checkpoint_every=args.checkpoint_every,
          interrupt_after_checkpoint=args.interrupt_after_checkpoint,output_root=args.output_root)
