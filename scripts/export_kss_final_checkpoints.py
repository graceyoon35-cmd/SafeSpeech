"""Copy final model weights to the desktop workspace; preserve full native states."""
import argparse
import json
from pathlib import Path
import shutil

from run_korean_melo_spec import save,sha256

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);args=p.parse_args()
results=json.loads((args.results/'kss_results.json').read_text())
trained=json.loads(Path(results['study']['training_summary']).read_text())
target=args.results/'final_checkpoints';target.mkdir(exist_ok=False)
records=[]
for row in trained['training']:
    source=Path(row['checkpoint']);report=json.loads(Path(row['report']).read_text())
    if sha256(source)!=report['checkpoint_sha256']:raise ValueError('Source model hash changed')
    destination=target/row['condition'];destination.mkdir()
    copied=destination/source.name;shutil.copyfile(source,copied)
    if sha256(copied)!=report['checkpoint_sha256']:raise ValueError('Desktop model copy mismatch')
    for name in ('config.json','training_report.json'):
        shutil.copyfile(Path(row['run'])/name,destination/name)
    records.append({'condition':row['condition'],'model':str(copied.resolve()),'sha256':report['checkpoint_sha256'],
        'native_full_resume_directory':report['checkpoint_directory'],
        'scope':'Desktop model-only copy; original optimizer/discriminator/RNG state remains in the native checkpoint directory.'})
save(target/'checkpoint_index.json',{'status':'THREE_FINAL_MODEL_COPIES_VERIFIED','models':records})
print(json.dumps({'directory':str(target.resolve()),'status':'THREE_FINAL_MODEL_COPIES_VERIFIED'}))
