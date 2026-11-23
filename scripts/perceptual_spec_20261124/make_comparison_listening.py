"""Exact-gain original/baseline/candidate comparisons for real listening."""
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from experiment_io import OUT,sha,save

def main():
    target=OUT/'listening_comparisons';target.mkdir(exist_ok=False)
    with (OUT/'quality/per_file.csv').open(encoding='utf-8-sig') as f:
        rows={r['profile']:r for r in csv.DictReader(f) if r['utterance_id']=='kss_hf_00078'}
    source=OUT/'quality/kss_hf_00078/original.wav'
    original,sr=sf.read(source,dtype='int16')
    baseline_path=Path(rows['eps2_iter10']['path']);assert sha(baseline_path)==rows['eps2_iter10']['sha256']
    baseline,br=sf.read(baseline_path,dtype='int16');assert sr==br==44100
    manifest=[];gap=np.zeros(26460,dtype=np.int16)
    for profile in ('eps1_iter1','eps1_iter3','eps1_iter5','mask_eps2_iter10_s05','mask_eps2_iter10_s2'):
        p=Path(rows[profile]['path']);assert sha(p)==rows[profile]['sha256']
        candidate,cr=sf.read(p,dtype='int16');assert sr==cr and candidate.shape==original.shape==baseline.shape
        output=target/('ABC_original_baseline_'+profile+'.wav')
        all_samples=np.concatenate([original,gap,baseline,gap,candidate])
        with output.open('xb') as f:sf.write(f,all_samples,sr,format='WAV',subtype='PCM_16')
        rr,decoded=wavfile.read(output);assert rr==sr and np.array_equal(decoded,all_samples)
        manifest.append(dict(candidate=profile,path=str(output),sha256=sha(output),
            original_sha256=sha(source),baseline_sha256=sha(baseline_path),candidate_sha256=sha(p),
            order='original,0.6s silence,baseline eps2/10,0.6s silence,candidate',
            gain=1,filters=False,human_review='PENDING',exact_samples_independently_verified=True))
    save(target/'manifest.json',manifest)
    with (target/'README.txt').open('x',encoding='utf-8') as f:
        f.write('순서: 원본 → 0.6초 무음 → 기존 SPEC ε=2/255·10회 → 0.6초 무음 → 파일명에 표시된 후보. 원래 샘플 진폭 그대로이며 필터·볼륨 보정 없음. 동일 재생 장치·음량에서 비교하세요. 음질 비교 파일이며 방어 성공 표시가 아닙니다. 별도 blind_listening/은 A/B/X 비교용입니다.\n')
    print('Exact-gain comparison WAVs:',len(manifest))

if __name__=='__main__':main()
