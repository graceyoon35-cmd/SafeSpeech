"""Source overlay archive for an uncommitted workspace; exclude audio and secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from run_korean_melo_spec import ROOT,git,save,sha256,TEAM,OFFICIAL,MODEL

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
files=set()
for pattern in ('scripts/*.py','scripts/*.ps1','docs/*.md','docs/templates/*.json','requirements*.txt','Review-KSS.cmd'):
    files.update(ROOT.glob(pattern))
manifest={'base_commit':git('rev-parse','HEAD').decode().strip(),'team_revision':TEAM,
    'melo_source_revision':OFFICIAL,'model_revision':MODEL,
    'scope':'Current source overlay on base commit, no Git commit created. Audio, checkpoints and account files excluded.',
    'files':{str(path.relative_to(ROOT)).replace('\\','/'):sha256(path) for path in sorted(files)}}
save(args.output/'source_overlay_manifest.json',manifest)
target=args.output/'source_overlay.zip'
with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
    for relative in manifest['files']:
        archive.write(ROOT/relative,relative)
    archive.write(args.output/'source_overlay_manifest.json','source_overlay_manifest.json')
with zipfile.ZipFile(target) as archive:
    if archive.testzip() is not None:raise ValueError('Source ZIP damaged')
    for relative,expected in manifest['files'].items():
        if hashlib.sha256(archive.read(relative)).hexdigest()!=expected:
            raise ValueError('Source ZIP checksum mismatch')
print(json.dumps({'zip':str(target.resolve()),'source_files':len(files),'sha256':sha256(target),'status':'PASS'}))
