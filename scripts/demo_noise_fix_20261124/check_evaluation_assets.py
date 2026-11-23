"""Verify archived metric model assets without downloads or runtime writes."""
import json
from pathlib import Path
from prepare_demo import HERE,sha,save

root=Path('/home/safespeech/models/melo-evaluation-20261118')
provenance=root/'model_provenance.json'
data=json.loads(provenance.read_text())
files={str(provenance):sha(provenance)}
for relative,expected in data['sha256'].items():
    p=root/relative
    actual=sha(p)
    if actual!=expected:raise ValueError(f'Metric model hash mismatch: {relative}')
    files[str(p)]=actual
save(HERE/'evaluation_assets_before.json',dict(status='PASS',sha256=files,downloads=False,existing_assets_modified=False))
print(json.dumps(dict(status='PASS',files=len(files),downloads=False)))
