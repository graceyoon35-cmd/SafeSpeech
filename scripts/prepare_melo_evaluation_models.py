"""Cache public speaker/ASR models and record pinned source hashes."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
import whisper

root=Path('/home/safespeech/models/melo-evaluation-20261118')
root.mkdir(parents=True,exist_ok=True)
repo='speechbrain/spkrec-ecapa-voxceleb'
record_path=root/'model_provenance.json'
if record_path.exists():
    revision=json.loads(record_path.read_text())['speechbrain_revision']
else:
    revision=HfApi().model_info(repo).sha
speaker=root/('ecapa-'+revision[:8])
snapshot_download(repo,revision=revision,local_dir=speaker,
    allow_patterns=['hyperparams.yaml','*.ckpt','label_encoder.txt','README.md'])
print('Speaker model cached',flush=True)
model=whisper.load_model('small',device='cpu',download_root=str(root/'whisper'))
del model
payload={'speechbrain_repo':repo,'speechbrain_revision':revision,'speechbrain_directory':str(speaker),
    'whisper_model':'small','whisper_directory':str(root/'whisper'),
    'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file() and '.cache' not in p.parts and p!=record_path}}
record_path.write_text(json.dumps(payload,indent=2))
print(json.dumps({'provenance':str(record_path),'speechbrain_revision':revision},indent=2))
