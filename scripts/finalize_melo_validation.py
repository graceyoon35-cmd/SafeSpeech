"""Post-run source snapshot and actual checkpoint/data integrity inventory."""
import argparse
import compileall
import json
from pathlib import Path
import shutil

from run_korean_melo_spec import ROOT,check_assets,save,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True)
    args=p.parse_args()
    if not compileall.compile_dir(ROOT/'scripts',quiet=1):
        raise ValueError('Python syntax check failed')
    args.output.mkdir(parents=True,exist_ok=False)
    _,_,assets=check_assets(args.assets)
    summary=json.loads(args.training.read_text())
    weights={}
    for run in summary['training']:
        report=json.loads(Path(run['report']).read_text())
        if sha256(run['checkpoint'])!=report['checkpoint_sha256']:
            raise ValueError('Checkpoint no longer matches training report')
        for name in ('G_3.pth','training_state.pth','training_report.json','config.json'):
            target=Path(run['run'])/name
            # G name follows actual completed step count rather than a fixed name.
            if name=='G_3.pth':
                target=Path(run['checkpoint'])
            weights[str(target)]={'bytes':target.stat().st_size,'sha256':sha256(target)}
    snapshot=args.output/'post_run_source_snapshot'
    sources={}
    for path in list((ROOT/'scripts').glob('*.py'))+list((ROOT/'scripts').glob('*.ps1'))+list((ROOT/'docs').glob('*.md'))+list(ROOT.glob('requirements*.txt')):
        dest=snapshot/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest);sources[str(path.relative_to(ROOT))]=sha256(dest)
    rows=json.loads((args.bundle/'spec_input.json').read_text())['utterances']
    for row in rows:
        if sha256(args.bundle/row['original_path'])!=row['original_sha256']:
            raise ValueError('KSS source hash changed')
    save(args.output/'integrity.json',{'status':'PASS','model_assets':assets,
        'checkpoints_and_training_state':weights,'source_snapshot_sha256':sources,
        'source_snapshot_scope':'Post-run final implementation snapshot, not a claim of a pre-run source snapshot.',
        'kss_original_wav_hashes_verified':len(rows),
        'kss_manifest_sha256':sha256(args.bundle/'spec_input.json'),
        'kss_research_permission_true_count':sum(r['research_use_permitted'] is True for r in rows),
        'kss_transcript_verified_true_count':sum(r['transcript_verified'] is True for r in rows)})
    print(json.dumps({'directory':str(args.output.resolve()),'status':'PASS'}))


if __name__=='__main__':
    main()
