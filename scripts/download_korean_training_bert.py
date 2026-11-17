"""Pinned public Korean BERT used by the official Melo Korean frontend."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

revision = '1779cc0982ada0216dd6de0dd4e86fb78201926d'
target = Path('/home/safespeech/models/bert-kor-base-' + revision[:8])
snapshot_download('kykim/bert-kor-base', revision=revision, local_dir=target,
                  allow_patterns=['config.json', 'vocab.txt', 'tokenizer*',
                                  'special_tokens_map.json', 'pytorch_model.bin', 'README.md'])
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in target.iterdir() if p.is_file()}
info=HfApi().model_info('kykim/bert-kor-base',revision=revision,files_metadata=True)
weight=next(f for f in info.siblings if f.rfilename=='pytorch_model.bin')
if weight.lfs.sha256!=hashes['pytorch_model.bin']:
    raise ValueError('Korean BERT remote LFS SHA-256 mismatch')
print(json.dumps({'revision': revision, 'directory': str(target), 'sha256': hashes,
                  'remote_lfs_sha256_verified':True}, indent=2))
