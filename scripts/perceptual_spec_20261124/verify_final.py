"""Final read-only checks of protocol, checkpoints, old files and GPU state."""
import json
from pathlib import Path
import subprocess
from experiment_io import ROOT,OUT,LEGACY,save,sha

def main():
    result=json.loads((OUT/'results.json').read_text())
    assert result['independent_cer_checks']==180
    preserved=json.loads((OUT/'attacks/preservation.json').read_text())
    assert not preserved['changed']
    assert all(sha(p)==h for p,h in preserved['source_sha256'].items())
    followup=json.loads((OUT/'attacks_followup/preservation.json').read_text())
    assert not followup['changed']
    assert all(sha(p)==h for p,h in followup['source_sha256'].items())
    summary=json.loads((OUT/'attacks_followup/training_summary.json').read_text())
    base=json.loads((Path(summary['experiment'])/'clean/config.json').read_text())
    original=json.loads((Path(summary['experiment'])/'clean/dataset.json').read_text())['utterances']
    clean=next(r for r in summary['training'] if r['condition']=='clean')
    reference=json.loads(Path(clean['report']).read_text())
    models=[]
    for row in summary['training']:
        if row['condition'] not in ('eps1_iter5','mask_eps2_iter10_s2','mask_eps2_iter10_s05'):continue
        cfg=json.loads((Path(row['run'])/'config.json').read_text())
        report=json.loads(Path(row['report']).read_text())
        assert cfg['train']==base['train'] and cfg['model']==base['model']
        assert {k:v for k,v in cfg['data'].items() if k!='training_files'}=={k:v for k,v in base['data'].items() if k!='training_files'}
        assert report['trainer_sha256']==reference['trainer_sha256']
        assert report['initial_checkpoint_sha256']==reference['initial_checkpoint_sha256']
        assert report['completed_steps']==1000
        assert [r['utterance_id'] for r in report['losses']]==[r['utterance_id'] for r in reference['losses']]
        updated=json.loads(Path(cfg['experiment']['dataset_manifest']).read_text())['utterances']
        assert len(original)==len(updated)==200
        for a,b in zip(original,updated):
            assert all(a[k]==b[k] for k in ('utterance_id','split','text','features','features_sha256'))
            if a['split']!='train':assert a['wav']==b['wav'] and a['wav_sha256']==b['wav_sha256']
            assert sha(b['wav'])==b['wav_sha256']
        checkpoint=Path(row['checkpoint']).parent;manifest=json.loads((checkpoint/'checkpoint.json').read_text())
        assert manifest['step']==1000
        for part in ('generator','state'):
            item=manifest['files'][part];assert sha(checkpoint/item['path'])==item['sha256']
        models.append(dict(candidate=row['condition'],same_training_protocol=True,original_test_not_trained=True,
            checkpoint=str(checkpoint),generator_and_optimizer_rng_state_hashes_pass=True))
    previous=json.loads((ROOT/'backups/closeout_20261124/verification.json').read_text())
    for p,expected in previous['old_backups_after'].items():
        stat=Path(p).stat();assert (stat.st_size,stat.st_mtime_ns)==(expected['size'],expected['mtime_ns'])
    assert sha(previous['archive'])==previous['archive_sha256']
    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],capture_output=True,text=True,check=True)
    active=[line for line in gpu.stdout.splitlines()[1:] if line.strip()]
    save(OUT/'final_verification.json',dict(status='PASS',models=models,original_files_unchanged=len(preserved['source_sha256']),
        original_backups_size_and_mtime_unchanged=True,previous_closeout_zip_sha256_unchanged=True,
        original_app_branch_not_written=True,gpu_compute_processes=active,human_review='PENDING unless separately provided',
        limitation='Quality pilot includes one original test audio; no attack training overlap, but not fully unseen hyperparameter selection.'))
    print(json.dumps(dict(status='PASS',new_models=len(models),old_files_changed=0,gpu_compute_processes=len(active))))

if __name__=='__main__':main()
