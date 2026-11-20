"""Create strictly epsilon-bounded PCM16 copies without replacing historical KSS."""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from audit_kss_final import digest, load, quality, table, write
from protect_wav import project_pcm16


def project(spec_run, output):
    pairs = load(Path(spec_run)/'handoff_manifest.json')['utterances']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for r in pairs:
        if r['iterations'] != 10:
            continue
        for role in ('clean','protected'):
            if digest(r[role+'_path']) != r[role+'_sha256']:
                raise ValueError('Source hash mismatch')
        a, sr = sf.read(r['clean_path'],dtype='float64')
        b, br = sf.read(r['protected_path'],dtype='float64')
        if sr != br or a.shape != b.shape:
            raise ValueError('Input pair mismatch')
        # Manifest epsilon is in normalized units; CLI helper takes /255 units.
        codes, changed = project_pcm16(a,b,r['epsilon']*255)
        path = output/(r['utterance_id']+'.wav')
        with path.open('xb') as stream:
            sf.write(stream,codes,sr,format='WAV',subtype='PCM_16')
        q = quality(r['clean_path'],path,r['epsilon'])
        if not q['strict_epsilon_pass']:
            raise ValueError('Saved copy exceeds strict epsilon')
        rows.append({'utterance_id':r['utterance_id'],'split':r['split'],
            'source':r['protected_path'],'source_sha256':r['protected_sha256'],
            'output':str(path.resolve()),'sha256':digest(path),'adjusted_samples':changed,**q})
    if not rows:
        raise ValueError('No SPEC_10 inputs')
    table(output/'quality.csv',rows)
    write(output/'summary.json',{'n':len(rows),
        'mean_stoi':float(np.mean([r['stoi'] for r in rows])),
        'mean_snr_db':float(np.mean([r['snr_db'] for r in rows])),
        'max_linf':max(r['linf'] for r in rows),
        'strict_epsilon_failures':sum(not r['strict_epsilon_pass'] for r in rows),
        'rail_samples':sum(r['protected_rail_samples'] for r in rows),
        'scope':'Saved PCM16 correction only. No model trained or SIM/CER measured on corrected copies.'})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    project(args.spec_run,args.output)
