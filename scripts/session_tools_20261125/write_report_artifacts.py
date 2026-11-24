"""Write playback manifest and output checksums. Does not generate or change audio."""
import ast
from pathlib import Path
import soundfile as sf
from segment_balance import OUT, sha, table, save

def main():
    for p in OUT.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))

    rows=[]
    for p in sorted((OUT/'listening').glob('*.wav')):
        info=sf.info(p)
        uid,variant=p.stem.split('__original_baseline_')
        rows.append(dict(utterance_id=uid,candidate=variant,path=str(p),sha256=sha(p),
            sample_rate=info.samplerate,subtype=info.subtype,channels=info.channels,seconds=info.duration,
            sequence='original;0.6s zero;eps2/255 iter10 baseline;0.6s zero;candidate',
            gain_normalization=False,human_audibility_unverified=True))
    assert len(rows)==9
    table(OUT/'listening_manifest.csv',rows)
    files=[dict(relative_path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
    save(OUT/'artifact_checksums.json',dict(files=files,self_not_included=True,
        no_new_training=True,defense_unmeasured=True,human_listening_unverified=True))
    print('Saved listening_manifest.csv and artifact_checksums.json:',len(files),'files,',sum(r['bytes'] for r in files),'bytes')


if __name__=='__main__':
    main()
