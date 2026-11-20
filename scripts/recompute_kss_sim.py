"""Independently re-encode preserved KSS evaluation WAVs for ECAPA SIM checks."""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

from audit_kss_final import digest, load, table, write


def recompute(evaluation, audit_path, output):
    evaluation, output = Path(evaluation), Path(output)
    metadata = load(evaluation)
    audit = load(audit_path)
    if audit['status'] != 'PASS':
        raise ValueError('Passing input audit required')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    model = SpeakerRecognition.from_hparams(
        source=metadata['evaluation_models']['speechbrain_directory'],
        savedir=str(output/'speaker_runtime'),run_opts={'device':'cuda:0'})

    def embedding(path):
        a, sr = sf.read(path,dtype='float32')
        if a.ndim!=1 or not np.isfinite(a).all():
            raise ValueError('Invalid evaluation waveform')
        x = torchaudio.functional.resample(torch.from_numpy(a),sr,16000)[None].cuda()
        return model.encode_batch(x,normalize=False)

    with (evaluation.parent/'per_file.csv').open(encoding='utf-8-sig',newline='') as stream:
        original = list(csv.DictReader(stream))
    result = []
    with torch.no_grad():
        refs = []
        for r in audit['sim_references']:
            if digest(r['wav'])!=r['sha256']:
                raise ValueError('SIM reference hash mismatch')
            refs.append(embedding(r['wav']))
        for r in original:
            if digest(r['output'])!=r['output_sha256']:
                raise ValueError('Evaluation audio hash mismatch')
            e = embedding(r['output'])
            value = float(np.mean([float(model.similarity(ref,e).reshape(-1)[0]) for ref in refs]))
            result.append({'condition':r['condition'],'sentence_id':r['sentence_id'],
                'original_sim':float(r['sim']),'recomputed_sim':value,
                'absolute_difference':abs(float(r['sim'])-value)})
    maximum = max(r['absolute_difference'] for r in result)
    table(output/'per_file.csv',result)
    write(output/'summary.json',{'status':'PASS' if maximum<1e-6 else 'FAIL',
        'n':len(result),'max_absolute_difference':maximum,
        'evaluation':str(evaluation.resolve()),'audit':str(Path(audit_path).resolve())})
    return maximum<1e-6


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    raise SystemExit(0 if recompute(args.evaluation,args.audit,args.output) else 1)
