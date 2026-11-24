"""Read saved CPU measurements/logs; no training, SPEC optimization or inference."""
import csv
import json
import math
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
from segment_balance import OUT, ROOT, OLD, sha, save, table


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def load(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def db(a):
    return 20*math.log10(max(float(a), 1e-15))


def main():
    summary=read(OUT/'analysis/summary.json')
    groups={r['condition']:r for r in summary}
    rows=load(OUT/'analysis/per_file.csv')
    cleanset=read(OLD/'runs/paired_melo_experiment_ong85l9i/clean/dataset.json')
    train=[r for r in cleanset['utterances'] if r['split']=='train']
    sources={}
    history=[]
    folders={
        'eps2_iter10':ROOT/'runs/demo_noise_fix_20261124/defense_attack_140/SPEC_eps2_iter10/protected_train',
        'perceptual05':ROOT/'runs/perceptual_spec_20261124/attacks_followup/mask_eps2_iter10_s05/protected_train',
        'perceptual2':ROOT/'runs/perceptual_spec_20261124/attacks/mask_eps2_iter10_s2/protected_train',
    }
    for item in train:
        starts=[]
        for label,folder in folders.items():
            p=folder/item['utterance_id']/'generation.json';sources[str(p)]=sha(p);g=read(p)
            assert sha(g['stats']['input'])==item['wav_sha256']
            assert g['settings']['seed']==52 and g['settings']['epsilon']==2 and g['settings']['iters']==10
            saved=g['quality'];assert sha(saved['path'])==saved['sha256']
            start=next(r for r in g['stats']['history'] if r['iter']==0)
            last=next(r for r in g['stats']['history'] if r['iter']==9)
            objective=lambda r:r['loss_mel']+10*(r['loss_nr']+r['loss_kl'])
            initial=objective(start);final=objective(last);starts.append(initial)
            history.append(dict(utterance_id=item['utterance_id'],condition=label,initial_surrogate=initial,
                iter9_preupdate_surrogate=final,reduction=initial-final,relative_reduction=(initial-final)/initial,
                meaning='SPEC generation surrogate before tenth update; not final saved-wave loss or cloning defense loss'))
        assert max(starts)-min(starts)<1e-4
    loss_summary=[]
    for label in folders:
        a=[r for r in history if r['condition']==label]
        loss_summary.append(dict(condition=label,n=len(a),**{key:float(np.mean([r[key] for r in a]))
            for key in ('initial_surrogate','iter9_preupdate_surrogate','reduction','relative_reduction')}))
    table(OUT/'analysis/old_objective_history.csv',history)
    save(OUT/'analysis/old_objective_review.json',dict(summary=loss_summary,
        new_candidates_objective='UNMEASURED',causal_proof=False,
        limitations='Observed old logs only. Iter9 is before tenth update, not saved-wave objective. SPEC surrogate is not cloning defense. No new loss/gradient/model inference was run.'))

    diagnostics=[]
    for uid in ('kss_hf_00078','kss_hf_00137','kss_hf_00347'):
        meta=read(OUT/'analysis'/(uid+'_segments.json'))
        x,sr=sf.read(meta['original'])
        selected=[r for r in rows if r['utterance_id']==uid and r['condition'] in
            ('eps2_iter10','silence_only_smooth','silence_floor10_smooth','voice_adaptive_smooth')]
        for r in selected:
            y,rate=sf.read(r['path']);assert rate==sr
            for s in meta['intervals']:
                lo=int(s['start_seconds']*sr);hi=min(len(x),int(s['end_seconds']*sr))
                if hi<=lo:continue
                rms=float(np.sqrt(np.mean((y[lo:hi]-x[lo:hi])**2)))
                original=float(np.sqrt(np.mean(x[lo:hi]**2)))
                diagnostics.append(dict(utterance_id=uid,condition=r['condition'],**s,
                    noise_rms=rms,noise_dbfs=db(rms),clean_rms=original,clean_dbfs=db(original),
                    residual_to_clean_db=db(rms)-db(original),interval_accuracy='20ms frames, 10ms grid; energy proxy'))
    table(OUT/'analysis/listening_segment_diagnostics.csv',diagnostics)

    deltas=[]
    base=groups['eps2_iter10']
    for label in ('silence_only_smooth','silence_floor10_smooth','voice_adaptive_smooth'):
        a=groups[label]
        rr=[r for r in rows if r['condition']==label]
        checks=dict(strict_pcm16_epsilon_pass=a['epsilon_failures']==0,zero_clipping=a['clipping']==0,
            mean_stoi_not_lower=a['stoi']>=base['stoi'],mean_silence_rms_lower=a['silence_noise_rms']<base['silence_noise_rms'],
            mean_weak_rms_not_higher=a['weak_noise_rms']<=base['weak_noise_rms'])
        assert all(checks.values())
        deltas.append(dict(condition=label,checks=checks,objective_preselection_pass=True,
            silence_rms_change_db=db(a['silence_noise_rms']/base['silence_noise_rms']),
            weak_rms_change_db=db(a['weak_noise_rms']/base['weak_noise_rms']),
            speech_residual_energy_retained_mean=a['voiced_residual_energy_retained'],
            max_pcm16_code_delta=max(float(r['linf'])*32768 for r in rr),
            worst_gate_modulation_step=max(float(r['max_gate_modulation_step']) for r in rr),
            worst_gate_modulation_step_dbfs=db(max(float(r['max_gate_modulation_step']) for r in rr)),
            per_file_stoi_lower_than_base=sum(float(r['stoi'])<float(next(b['stoi'] for b in rows if b['utterance_id']==r['utterance_id'] and b['condition']=='eps2_iter10')) for r in rr),
            note='Objective screen only; not a human audibility/quality approval or defense evidence.'))
    save(OUT/'analysis/preselection.json',dict(candidates=deltas,average_definition='Arithmetic utterance means; RMS reduction dB = 20 log10(ratio of mean RMS), not mean per-utterance dB.',
        limits='Strong-speech proxy uses clean energy above P90-15dB. Energy retention is not loss or defense retention.'))

    # Read actual archived evaluation metrics rather than attributing them to new WAVs.
    evals=list((ROOT/'runs/perceptual_spec_20261124/attacks_followup/runs').glob('melo_smoke_evaluation_*/evaluation.json'))
    assert len(evals)==1
    sources[str(evals[0])]=sha(evals[0]);evaluation=read(evals[0]);assert evaluation['status']=='EVALUATION_PASS'
    archived=evaluation['measurements']
    save(OUT/'analysis/archived_defense_reference.json',dict(source=str(evals[0]),sha256=sha(evals[0]),
        measurements=archived,new_candidates_have_no_sim_or_cer=True,
        limitations='Single speaker/seed/architecture, reused test prompts and adaptive prior follow-up; exploratory comparisons, not unseen validation.'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bands=((0,200),(200,1000),(1000,4000),(4000,8000),(8000,16000),(16000,22051))
    spectrum=[]
    for label in groups:
        a=[r for r in rows if r['condition']==label]
        for region in ('all','silence','weak','voiced'):
            for lo,hi in bands:
                key=f'{region+"_" if region!="all" else ""}band_{lo}_{hi}_power'
                values=[float(r[key]) for r in a if r.get(key)]
                spectrum.append(dict(condition=label,region=region,band_hz=f'{lo}-{hi}',
                    n=len(values),mean_power=float(np.mean(values)) if values else None,
                    mean_power_dbfs=10*math.log10(max(float(np.mean(values)),1e-30)) if values else None))
    table(OUT/'analysis/band_energy.csv',spectrum)
    fig,axes=plt.subplots(1,2,figsize=(13,4.6))
    for ax,region in zip(axes,('all','silence')):
        for label in ('eps8_iter10_strict','eps2_iter10','perceptual05','eps1_iter5','silence_only_smooth','voice_adaptive_smooth'):
            a=[r for r in spectrum if r['condition']==label and r['region']==region]
            ax.plot(range(len(bands)),[r['mean_power_dbfs'] for r in a],marker='.',label=label)
        ax.set_xticks(range(len(bands)),['0-.2','.2-1','1-4','4-8','8-16','16-22.05'])
        ax.set(xlabel='Frequency band (kHz)',ylabel='Mean band power (dBFS)',title=region+' residual energy')
        ax.grid(alpha=.25)
    axes[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'analysis/band_energy.png',dpi=150);plt.close(fig)

    # Bind future inputs/settings without importing the GPU trainer.
    plan=read(OUT/'future_attack_plan.json');cfgs=[]
    for label in plan['selected_candidates']:
        cfgpath=OUT/'plans'/label/'config.json';cfg=read(cfgpath);ds=read(cfg['experiment']['dataset_manifest'])
        assert cfg['train']['seed']==52
        assert len([r for r in ds['utterances'] if r['split']=='train'])==140
        for r,old in zip(ds['utterances'],cleanset['utterances']):
            assert r['utterance_id']==old['utterance_id'] and r['split']==old['split'] and r['features']==old['features']
            assert sha(r['wav'])==r['wav_sha256'] and sha(r['features'])==r['features_sha256']
            if r['split']!='train':assert r==old
        initial=cfg['experiment']['generator_initialization'];initial_hash=sha(initial)
        assert initial_hash=='48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0'
        cfgs.append(dict(candidate=label,config=str(cfgpath),config_sha256=sha(cfgpath),
            dataset_sha256=sha(cfg['experiment']['dataset_manifest']),seed=52,
            initial_checkpoint=initial,initial_sha256=initial_hash,
            target_output=str(OUT/'future_attacks'/label),
            future_only_command=['/home/safespeech/.venvs/melo-training-20261118/bin/python','-B',plan['training']['trainer'],
                '--config',str(cfgpath),'--assets','/home/safespeech/models/melo-spec-pinned-20261118',
                '--steps','1000','--checkpoint-every','1000','--output-root',str(OUT/'future_attacks'/label)]))
    save(OUT/'plans/verified_inputs.json',dict(status='PLAN_ONLY_NOT_EXECUTED',candidates=cfgs,
        quality_preselection=str(OUT/'analysis/preselection.json'),
        evaluation='Same archived evaluator/metric weights, five clean ECAPA references, Whisper small ko temp0, seed2026+i; exact old 30 prompts. Add independent held-out human listening and fresh prompts before generalizing.',
        training_not_started=True))

    finish_audit(sources)
    print(json.dumps(dict(old_loss=loss_summary,preselection=deltas,archived=[{k:r[k] for k in ('condition','mean_sim','mean_cer')} for r in archived],
        baseline_silence_dbfs=db(base['silence_noise_rms'])),ensure_ascii=False,indent=2))


def finish_audit(sources=None):
    # Windows Git owns this worktree's Windows absolute gitdir; native WSL Git cannot resolve it.
    rows=load(OUT/'analysis/per_file.csv')
    if sources is None:
        sources={}
        for folder in (ROOT/'runs/demo_noise_fix_20261124/defense_attack_140/SPEC_eps2_iter10/protected_train',
                       ROOT/'runs/perceptual_spec_20261124/attacks_followup/mask_eps2_iter10_s05/protected_train',
                       ROOT/'runs/perceptual_spec_20261124/attacks/mask_eps2_iter10_s2/protected_train'):
            for p in folder.glob('*/generation.json'):sources[str(p)]=sha(p)
        for p in (ROOT/'runs/perceptual_spec_20261124/attacks_followup/runs').glob('melo_smoke_evaluation_*/evaluation.json'):
            sources[str(p)]=sha(p)
    existing=read(OUT/'preservation.json')['original_sha256']
    existing.update(sources)
    changed=[p for p,h in existing.items() if sha(p)!=h];assert not changed
    mismatched=[r['path'] for r in rows if sha(r['path'])!=r['sha256']];assert not mismatched
    winroot=subprocess.check_output(['wslpath','-w',str(ROOT)],text=True).strip()
    git=lambda *args:subprocess.check_output(['/mnt/c/Program Files/Git/cmd/git.exe','-C',winroot,*args],text=True).strip()
    branch=git('branch','--show-current');head=git('rev-parse','HEAD');status=git('status','--porcelain')
    assert branch=='research/perceptual-spec-20261124' and head=='954a06cde13065074328bc9476eaf8238845d1ea' and not status
    save(OUT/'final_audit.json',dict(watched_sources=len(existing),changed=changed,
        reviewed_sources_sha256=sources,candidate_and_source_csv_hash_mismatches=mismatched,
        candidate_wavs=len(list((OUT/'candidates').glob('*/*.wav'))),listening_wavs=len(list((OUT/'listening').glob('*.wav'))),
        generated_wavs_bytes=sum(p.stat().st_size for p in (OUT/'candidates').glob('*/*.wav')),
        branch=branch,head=head,git_status=status,no_commit_push=True,no_new_training_or_gpu_inference=True,
        human_listening_unverified=True,new_defense_unmeasured=True))
    print(json.dumps(dict(final_audit='PASS',watched_sources=len(existing),changed=changed,branch=branch,head=head),ensure_ascii=False))


if __name__=='__main__':main()
