"""CPU-only closeout: checkpoint inspection, exclusive ZIP backup and shutdown audit.
Never starts training/evaluation, deletes files, or overwrites an existing ZIP.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/demo_noise_fix_20261124'
OUT=ROOT/'backups/closeout_20261124'
ARCHIVE=OUT/'SafeSpeech_demo_noise_20261124.zip'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())

def checkpoints():
    import torch
    summary=json.loads((RUN/'defense_attack_140/training_summary.json').read_text())
    records=[]
    for row in summary['training']:
        if not row['condition'].startswith('SPEC_eps'):continue
        folder=Path(row['checkpoint']).parent
        manifest=json.loads((folder/'checkpoint.json').read_text())
        assert manifest['step']==1000 and manifest['schema_version']==2
        for kind in ('generator','state'):
            item=manifest['files'][kind]
            assert Path(item['path']).name==item['path']
            assert sha(folder/item['path'])==item['sha256']
        state=torch.load(folder/manifest['files']['state']['path'],map_location='cpu',weights_only=True)
        assert state['step']==1000 and len(state['history'])==1000
        assert state['discriminator'] and state['duration_discriminator']
        assert len(state['optimizers'])==3 and all(o['state'] and o['param_groups'] for o in state['optimizers'])
        rng=state['rng'];assert all(k in rng for k in ('torch','cuda','numpy','python'))
        assert rng['torch'].numel()>0 and rng['cuda'] is not None and rng['cuda'].numel()>0
        records.append(dict(condition=row['condition'],directory=str(folder),step=state['step'],
            files=manifest['files'],optimizer_count=3,optimizer_states_nonempty=True,
            torch_rng_bytes=rng['torch'].numel(),cuda_rng_bytes=rng['cuda'].numel(),
            numpy_rng_saved=True,python_rng_saved=True,discriminators_saved=True))
        del state;gc.collect()
    assert len(records)==2
    save(OUT/'checkpoint_inventory.json',dict(status='PASS',map_location='cpu',training_started=False,checkpoints=records))
    print(json.dumps(dict(status='PASS',complete_checkpoints=len(records),optimizer_and_rng=True)),flush=True)

def old_backup_stats():
    base=ROOT.parent/'SafeSpeech-backups'
    return {str(p):dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        for p in (base/'SafeSpeech_20261118_030155.zip',base/'SafeSpeech_SPEC_final_20261120_215028.zip')}

def sources():
    paths=set()
    directories=[ROOT/'runs/demo_noise_fix_20261124',ROOT/'runs/audio_quality_diagnosis_20261123',
        ROOT/'scripts/demo_noise_fix_20261124',ROOT/'scripts/audio_quality_diagnosis_20261123']
    for directory in directories:
        for p in directory.rglob('*'):
            if p.is_file():paths.add(p)
    for relative in ('.gitignore','.gitattributes','RESUME_DEMO_20261124_KO.md','scripts/spec_demo_profile.py','scripts/spec_demo_app.py',
        'scripts/launch_spec_demo.py','config/spec_demo.template.json','patches/spec_demo_ui_20261124.patch',
        'tools/closeout_demo_20261124.py','backups/closeout_20261124/checkpoint_inventory.json','backups/closeout_20261124/git_receipt.json'):
        p=ROOT/relative
        if not p.is_file():raise FileNotFoundError(p)
        paths.add(p)
    return sorted(paths)

def backup():
    assert json.loads((OUT/'checkpoint_inventory.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'git_receipt.json').read_text())['status']=='PUSHED'
    old_before=old_backup_stats()
    files=sources();total=sum(p.stat().st_size for p in files)
    if shutil.disk_usage(OUT).free<total*1.1+1024**3:raise RuntimeError('Insufficient free space')
    inventory=[]
    for i,p in enumerate(files):
        stat=p.stat()
        inventory.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
            sha256=sha(p),dereferenced_symlink=p.is_symlink()))
        if i%500==0:print('inventory',i,len(files),flush=True)
    manifest=dict(schema_version=1,date_kst='2026-11-24',scope='All new local diagnosis/demo artifacts, two new generator/state checkpoints, selected source and resume document. Prior backups are preserved separately.',
        files=inventory,source_bytes=total,old_backups_before=old_before)
    save(OUT/'source_manifest.json',manifest)
    manifest_bytes=(OUT/'source_manifest.json').read_bytes()
    with ARCHIVE.open('xb') as raw:
        with zipfile.ZipFile(raw,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
            for i,item in enumerate(inventory):
                archive.write(ROOT/item['path'],arcname=item['path'])
                if i%500==0:print('archive',i,len(inventory),flush=True)
            archive.writestr('_closeout/source_manifest.json',manifest_bytes)
        raw.flush();os.fsync(raw.fileno())
    zip_sha=sha(ARCHIVE)
    with (OUT/(ARCHIVE.name+'.sha256')).open('x',encoding='ascii') as f:
        f.write(zip_sha+'  '+ARCHIVE.name+'\n');f.flush();os.fsync(f.fileno())
    print('full CRC check',flush=True)
    verified=[]
    with zipfile.ZipFile(ARCHIVE,'r') as archive:
        assert len(archive.infolist())==len(inventory)+1
        bad=archive.testzip()
        if bad is not None:raise RuntimeError('ZIP CRC mismatch: '+bad)
        for i,item in enumerate(inventory):
            h=hashlib.sha256()
            with archive.open(item['path'],'r') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==item['sha256'],item['path']
            assert sha(ROOT/item['path'])==item['sha256'],item['path']
            assert (ROOT/item['path']).stat().st_size==item['bytes']
            verified.append(item['path'])
            if i%500==0:print('member SHA / source recheck',i,len(inventory),flush=True)
        assert archive.read('_closeout/source_manifest.json')==manifest_bytes
    assert sha(ARCHIVE)==zip_sha
    old_after=old_backup_stats();assert old_after==old_before
    save(OUT/'verification.json',dict(status='PASS',archive=str(ARCHIVE),archive_bytes=ARCHIVE.stat().st_size,
        archive_sha256=zip_sha,archive_sha256_repeat_matches=True,zip64=True,compression='ZIP_STORED',
        crc_all_entries_pass=True,crc_entries=len(inventory)+1,source_and_member_sha256_matches=len(verified),
        source_files_unchanged=True,embedded_manifest_matches=True,checkpoint_generator_and_state_files=4,
        old_backup_size_and_mtime_unchanged=True,old_backups_after=old_after,
        old_backup_contents_not_written=True,verification_scope='Every archived source file hashed before and after; every ZIP entry CRC checked; each data entry SHA256 matched source; whole archive SHA256 computed twice.'))
    print(json.dumps(dict(status='PASS',archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=zip_sha,files=len(verified))),flush=True)

def shutdown_audit():
    active=[]
    names={'train_melo_bounded.py','defense_compare.py','evaluate_paired_melo_smoke.py','prepare_demo.py','diagnose.py',
        'test_demo_app.py','test_wslg_startup.py','launch_spec_demo.py'}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            args=(p/'cmdline').read_bytes().split(b'\0')
            decoded=[a.decode(errors='replace') for a in args if a]
            if any(Path(a).name in names for a in decoded):active.append(dict(pid=int(p.name),command=decoded))
        except (PermissionError,FileNotFoundError,ProcessLookupError):pass
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],
        capture_output=True,text=True,check=True)
    gpu=[x for x in result.stdout.splitlines()[1:] if x.strip()]
    verify=json.loads((OUT/'verification.json').read_text());receipt=json.loads((OUT/'git_receipt.json').read_text())
    safe=not active and not gpu and verify['status']=='PASS' and receipt['status']=='PUSHED'
    save(OUT/'safe_shutdown.json',dict(status='SAFE_TO_SHUT_DOWN_FOR_SAFESPEECH' if safe else 'REQUIRES_REVIEW',
        date_kst='2026-11-24',active_safespeech_jobs=active,gpu_compute_processes=gpu,
        nvidia_smi_output=result.stdout,backup_verified=verify['status']=='PASS',git_push_verified=receipt['status']=='PUSHED',
        no_new_training_or_experiment_started=True,scope='SafeSpeech work only; this command does not power off the PC or close unrelated applications.'))
    print(json.dumps(dict(safe_for_safespeech=safe,active_jobs=len(active),gpu_compute_processes=len(gpu))),flush=True)
    if not safe:raise RuntimeError('Shutdown audit needs review')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('checkpoints','backup','shutdown'))
    action=p.parse_args().action
    {'checkpoints':checkpoints,'backup':backup,'shutdown':shutdown_audit}[action]()
