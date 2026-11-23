"""Separate output and read-only legacy assets for the perceptual SPEC study."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/perceptual_spec_20261124'
LEGACY=ROOT.parent/'SafeSpeech-desktop-integration-20261118'
ASSETS=Path('/home/safespeech/models/melo-spec-pinned-20261118')
sys.path.insert(0,str(LEGACY/'scripts'))
os.environ['PYTHONDONTWRITEBYTECODE']='1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def table(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

PROFILES={f'eps{e}_iter{i}':dict(epsilon=float(e),iterations=i,strength=0.) for e in (1,2) for i in (1,3,5,10)}
PROFILES.update(mask_eps2_iter10_s05=dict(epsilon=2.,iterations=10,strength=.5),
                mask_eps2_iter10_s2=dict(epsilon=2.,iterations=10,strength=2.),
                mask_eps2_iter5_s05=dict(epsilon=2.,iterations=5,strength=.5))
