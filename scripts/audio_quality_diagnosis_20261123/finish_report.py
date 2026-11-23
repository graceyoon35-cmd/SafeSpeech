"""Create listening artifacts, Korean report, and independently verify WAV copies."""
import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(os.environ.get('SAFESPEECH_DIAGNOSIS_RUN',str(ROOT/'runs/audio_quality_diagnosis_20261123'))).resolve()
def read(name): return json.loads((OUT/name).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)

outcome=read('outcome.json')
trio=outcome['trio']; compare=read('app_comparison.json')
folder=OUT/'listening/app_trio'; folder.mkdir()
waves=[]
for i,r in enumerate(trio):
    dest=folder/f"{i:02d}_{r['role']}.wav"
    shutil.copyfile(r['path'],dest)
    assert sha(dest)==r['sha256']
    waves.append(sf.read(dest)[0])
with (folder/'ABC_original_direct_app.wav').open('xb') as f:
    sf.write(f,np.concatenate([waves[0],np.zeros(26460),waves[1],np.zeros(26460),waves[2]]),44100,format='WAV',subtype='PCM_16')
with (folder/'README.txt').open('x',encoding='utf-8') as f:
    f.write('ABC 순서: 원본 → 0.6초 무음 → SPEC 직접 출력 → 0.6초 무음 → 앱 저장본. 세 구간 모두 원래 진폭 그대로. 직접 출력과 앱 저장본은 동일 샘플입니다. 2026-11-20 연동 테스트 저장본이며 이번 사용자 신고 파일과의 동일성은 미확인입니다.\n')

# Independent PCM decoder and exact listening segment verification.
checks=[]
for p in (OUT/'listening').rglob('*.wav'):
    sr,codes=wavfile.read(p)
    a,ar=sf.read(p,dtype='float64')
    decoded=codes.astype('float64')/32768 if codes.dtype==np.int16 else codes.astype('float64')
    assert sr==ar and np.array_equal(a,decoded),str(p)
    checks.append(dict(path=str(p),sha256=sha(p),independent_decode_exact=True))
for r in read('listening/manifest.json'):
    a,sr=sf.read(r['original']); b,br=sf.read(r['candidate']); ab,abr=sf.read(r['ab']); e,er=sf.read(r['residual'])
    assert sr==br==abr==er
    assert np.array_equal(ab[:len(a)],a) and np.array_equal(ab[len(a)+round(.6*sr):],b)
    assert np.count_nonzero(ab[len(a):len(a)+round(.6*sr)])==0
    assert np.array_equal(e,b-a)
write('listening_verification.json',dict(files=len(checks),all_pass=True,checks=checks,ab_order_and_gain_exact=True,residuals_exact=True))

historical=outcome['historical']; summary=outcome['candidates']; app=trio[1]
lines=['SafeSpeech 잡음 원인 진단 및 음질 후보 보고서 — 2026-11-23 KST',
'', '판정',
'확인된 연동 테스트에서는 SPEC 출력 자체에 큰 차이 신호가 존재한다. 앱 저장본은 직접 출력과 바이트·샘플이 모두 동일하다. 저장 중 재인코딩·추가 잡음이 발생했다는 증거는 없다.',
'이번에 사용자가 들은 파일은 특정되지 않았다. 본 결과는 2026-11-20 앱 실제 저장 모듈의 연동 테스트 산출물에 한정한다. 현재 앱 전체 GUI·스피커 출력·OS 믹서·장치 효과는 미검증이므로 앱 전체의 무결성을 보장하지 않는다.',
'', '원본 / SPEC 직접 출력 / 앱 저장본 비교',
'모두 44,100Hz, PCM16(16bit), mono, 141,123샘플, 3.200068초. SNR과 STOI의 기준은 같은 원본. 원본 자기 비교의 SNR은 +무한대이며 JSON에서는 null과 상태 문자열로 표현한다.']
for r in trio:
    snr=f"{r['snr_db']:.6f}dB" if r['snr_db'] is not None else '+무한대'
    lines.append(f"{r['role']}: peak {r['peak']:.9f}, RMS {r['rms']:.9f}, SNR {snr}, STOI {r['stoi']:.9f}, rail/clipping {r['rail_samples']}, 범위 초과 {r['out_of_range_samples']}")
lines += [f"직접 출력/앱 저장 공통 SHA256: {compare['sha256_a']}",
'다른 샘플 0, 최대 샘플 차이 0. app_trio.csv와 app_comparison.json에 경로·해시·수치 기록.',
'', '잡음 발생 기전과 저장 보정의 한계',
'고정 backend의 SPEC 모드는 mel 재구성·무작위 잡음 기준 NR·KL 목적함수로 sign-gradient를 누적하며, 사람 청취 지표 STOI/STFT 항은 SafeSpeech 분기에만 있다. SPEC에는 청각 마스킹이나 상대 발화 에너지에 따른 제약이 없다.',
'기본 step=ε/10이다. 1회는 최대 ε/10, 10회는 최대 ε까지 변화할 수 있어 반복 횟수 비교는 누적 진폭 차이도 포함한다. ε는 최대 샘플 차이 제한일 뿐 자연스러움 보장이 아니다.',
f"연동 사례의 차이 신호 RMS={app['delta_rms']:.9f}, 원본 RMS 대비 {100*app['delta_rms']/trio[0]['rms']:.2f}%. 조용한 20% 프레임의 차이 RMS={app['quiet20_delta_rms']:.9f}. 차이 에너지의 {100*app['delta_energy_above_8khz']:.2f}%가 8kHz 이상에 있다.",
'조용한 프레임은 원본의 20ms 프레임 에너지 하위 20%로 정의한 상대 구간이며, 실제 무음 검출이나 청취 판단을 뜻하지 않는다. 이 측정은 말소리 밖에서도 차이 신호가 남는다는 근거다. 원본 자체의 녹음 잡음과 섭동 차이 신호를 구분했다.',
'정수 ε 보정 전 optimized.wav와 protected.wav는 17,893샘플이 1 LSB(1/32768)만큼 다르다. SNR 11.594798→11.596944dB, STOI 0.895423→0.895463. 큰 잡음의 해결책으로 볼 수 없는 규모다.',
'', '기존 파일 재측정: 각 조건 200개']
for r in historical:
    snr=f"{r['mean_snr_db']:.6f}" if r['mean_snr_db'] is not None else '+무한대'
    lines.append(f"{r['condition']}: n={r['n']}, 평균 SNR {snr}dB / STOI {r['mean_stoi']:.9f}, rail {r['rail_samples']}, 최대 delta {r['max_linf']:.12f}")
lines += ['historical_files.csv에 원본 200개·기존 SPEC_1 200개·기존 SPEC_10 200개·보정 SPEC_10 200개의 개별 sr/bit/channels/peak/RMS/SNR/STOI/rail/해시를 기록했다.',
'', '새 SPEC 최적화 후보 (ε /255 단위, 각 n=5)',
'보정본 200개의 SNR 정렬 순위 0·49·99·149·199에 해당하는 입력 5개를 새 결과 생성 전에 선정했다. 같은 5개·같은 모델·seed52·기본 step=ε/10로 6조건 30개를 새로 최적화했다. 기존 섭동의 단순 축소 사본이 아니다. 무작위 표본이나 방어 평가용 독립 test set으로 간주하지 않는다.']
for r in summary:
    lines.append(f"ε={int(r['epsilon'])}/255, {r['iterations']}회: 평균 SNR {r['mean_snr_db']:.4f}dB, STOI {r['mean_stoi']:.6f}, 최저 STOI {r['min_stoi']:.6f}, 차이 RMS {r['mean_delta_rms']:.9f}, rail {r['clipping_samples']}")
lines += ['모든 후보는 44.1kHz PCM16 mono, 원래 길이·원래 진폭을 유지한다. 출력별 엄격한 ε 및 rail 0을 검사했다. 볼륨 감소·필터·자동 정렬·리샘플·자동 자르기·대규모 재학습은 수행하지 않았다.',
'GPU 기본 모드는 기존에 비결정론성이 관찰되었으므로 같은 seed 재실행의 바이트 동일성을 보장하지 않는다. 결과 WAV·설정·backend 소스·모델 해시를 보관했다.',
'', '청취 및 음질 개선 가능성',
'listening/app_trio/ABC_original_direct_app.wav: 원본→0.6초 무음→직접 출력→0.6초 무음→앱 저장본.',
'listening/<발화ID>/AB_original_then_*.wav: 원본→0.6초 무음→조건별 보호본. 각 구간의 진폭을 원본 그대로 유지했으며 독립 디코더 및 샘플 비교로 이를 검증했다.',
'원본·기존 보정본·새 후보 개별 WAV와 원래 크기 차이 신호 residual_*_FLOAT.wav도 제공한다. 차이 신호는 진단용이고 보호 음성의 청취 대체물이 아니다.',
'객관적 개선 여부는 candidates.csv와 candidate_summary.json으로 판단한다. STOI는 명료도 지표이며 자연스러움·불가청 보장이 아니다. 사람 청취 검수는 미수행(PENDING)이며 listening/human_ratings.csv에 동일 장치·동일 재생 음량으로 잡음·자연스러움·선호도를 기록해야 한다.',
'작은 ε·1회 후보를 먼저 청취할 수 있으나 출시 품질 통과나 최적 보호 설정으로 확정하지 않는다. 조용한 구간 에너지·청각적 손실을 고려한 별도 최적화는 후속 후보이며 검증 없이 필터를 성공 처리하지 않는다.',
'', '방어 성능 영향',
'섭동 크기·반복을 낮추면 음질 개선과 함께 방어 효과도 감소할 가능성이 있다. 그 크기나 실제 감소 여부는 현재 데이터로 판단할 수 없다. 대리모델 loss 변화는 실제 음성 복제 방어 입증이 아니다.',
'이번 보정본과 새 ε/반복 후보로 학습한 복제 모델의 SIM·CER/WER는 미측정이다. 기존 보호 WAV로 학습한 모델의 과거 결과를 새 후보의 성능으로 재사용하지 않는다.',
'청취로 후보를 좁힌 후, 원본/각 후보에 동일한 화자·분할·학습예산·새 평가 절차를 적용한 별도 복제 실험이 필요하다. 음질·명료도와 복제 음성 SIM·CER/WER를 나란히 보고해야 한다. 이번 작업에서는 해당 학습을 시작하지 않았다.',
'', '보존과 재현',
'기존 실험·모델·체크포인트·백업·앱팀 브랜치에 대한 쓰기를 수행하지 않았다. 읽은 파일 해시의 실행 전후 변화는 preservation.json에서 확인한다. 모든 기존 자산 전체를 다시 해시했다고 주장하지 않는다.',
'새 산출물은 현재 저장소 runs/audio_quality_diagnosis_20261123/에만 있다. diagnose.py, finish_report.py를 포함한다. 기존 디렉터리에는 쓰지 않도록 Python -B를 사용했다.',
"환경: /home/safespeech/.venvs/melo-training-20261118/bin/python; NVIDIA RTX 5060 Ti 8151MiB. 기존 모델 로딩과 섭동 최적화만 실행.",
'재실행은 diagnose.py를 새 결과 폴더로 복사한 후 기존 결과 없는 경로에서 실행한다. OLD 경로는 기존 연구 저장소, OUT은 스크립트 위치다. 기존 결과 덮어쓰기는 x/xb와 mkdir(exist_ok=False)로 거부한다.',
'', '남은 확인',
'이번 실제 신고 파일의 3종 경로 확보 후 동일 비교, 현재 앱 GUI/오디오 장치 재생 경로 확인, 사람 청취 평가, 선정 후보의 별도 방어 성능 검증.']
with (OUT/'REPORT_KO.txt').open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
print('\n'.join(lines[:18]))
print('Verification:',len(checks),'WAVs PASS')
