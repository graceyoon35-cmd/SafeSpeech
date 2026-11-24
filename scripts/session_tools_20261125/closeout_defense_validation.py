"""Record completed-run process/resource state and compact artifact inventory."""
import json
import shutil
import subprocess
from datetime import datetime,timezone
from run_defense_validation import TARGET,ROOT,sha,save

def main():
    assert json.loads((TARGET/'final_verification.json').read_text())['status']=='PASS'
    ps=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
    active=[line for line in ps.splitlines() if any(name in line for name in (
        'train_melo_bounded.py','evaluate_paired_melo_smoke.py','run_defense_validation.py','summarize_defense_validation.py'))]
    assert not active,active
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu','--format=csv'],text=True)
    files=[]
    for p in sorted(TARGET.rglob('*')):
        if p.is_file() and p.suffix not in ('.pth','.pt'):
            files.append(dict(path=str(p.relative_to(TARGET)),bytes=p.stat().st_size,sha256=sha(p)))
    # Large generator and optimizer/RNG files were already verified by checkpoint.json
    # and unchanged verify_checkpoint(), avoiding unnecessary repeated multi-GB reads.
    checkpoints=[json.loads(p.read_text()) for p in TARGET.glob('checkpoints/*/checkpoints/step_001000/checkpoint.json')]
    assert len(checkpoints)==2
    save(TARGET/'postrun_state.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        nvidia_smi=gpu,active_training_evaluation_processes=active,disk_free_bytes=shutil.disk_usage(ROOT).free,
        small_artifacts_sha256=files,committed_checkpoints=checkpoints,
        existing_results_preserved=True,no_git_commit_push=True))
    print(json.dumps(dict(active_training_or_evaluation=active,checkpoints=len(checkpoints),artifact_files=len(files),nvidia_smi=gpu),ensure_ascii=False))


if __name__=='__main__':
    main()
