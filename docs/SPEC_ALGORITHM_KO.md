# SPEC 알고리즘과 저잡음 연구

2026-11-26까지 수행한 기존 실험을 통합한다. 이 문서 작업에서 음성 생성·학습·설치·새 방어 평가는 하지 않았다. 공격 지표는 [최종 보고서](SPEC_FINAL_REPORT_20261120_KO.md), 실행은 [인계](README_SPEC_HANDOFF_20261120_KO.md)를 따른다.

## 기존 SPEC와 PCM16 제약

고정 Melo backend의 생성 목적은 `L_SPEC = L_mel + 10*(L_noise_reconstruction + L_KL)`이며 작을수록 생성 목적 최적화가 강하다. 모델 가중치는 고정하고 음성 입력의 부호 gradient로 갱신한다. 이름이 비슷한 논문의 SafeSpeech 전체 모드와 구분한다. SafeSpeech 모드의 미분용 NegSTOILoss를 최종 WAV의 `pystoi.stoi(..., extended=False)` 점수로 대신하지 않으며 누락 시 조용히 SPEC로 대체하지 않는다.

ε는 진폭 상한이지 지각적 품질 보장이 아니다. CLI `--epsilon 2`=2/255이며 PCM16 출력은 원본 대비 정수차 `floor((2/255)*32768)=257`code 이하로 제한한다. ε8은1028code다. 과거 SPEC_1은 ε8·1회이고 ε1·5회와 다르다.10회 기준 기본 step은 ε/10이다. 원본 RMS를 낮추거나 출력 전체를 필터링해 잡음을 가리는 방법으로 성공 판정하지 않는다.

초기 BERT-VITS2 영어1.02004초/FLOAT 보호본의1·10회 STOI는0.96124680/0.77541912였다. Mac CPU의 실제 Melo 영어 결과는0.96105714/0.78547233,11/10 한국어 합성 입력은0.99616034/0.87574336이었다. 실제 가중치·gradient·저장 경로 확인이며 사람 KSS 복제 방어가 아니다. Mock 계약, NONHUMAN 합성, 실제 녹음 공격을 구분한다. GPU 고정 seed는 바이트 재현을 보장하지 않으며 강제 결정론 실패도 보존한다.

## 무음 억제형의 개선과 남은 발화 잡음

원본20ms RMS/10ms 간격의 에너지 대리 VAD로 무음·약한·강한 구간을 나눴다. 사람 검수 VAD가 아니다. `silence_only_smooth`는 원본 기반 활동 envelope로 무음 섭동을 낮추고 부드럽게 전환한다. `voice_adaptive_smooth`는 약한 발화까지 제어하지만 실제 공격 방어가 크게 약해졌다.

같은 train140개에서 무음 억제형은 전체 SNR+2.99577dB, 무음 섭동 RMS−8.29366dB였다. 그러나 강한 발화140개·9,399,915샘플의 보호 PCM은 기존 ε2와 같았고 약한 발화 RMS 감소도 약0.3%였다. **전체 SNR 개선을 발화 중 잡음 개선으로 해석하지 않는다.** 약한/강한 구간의 SNR 평균8.9507/28.0098dB와 비슷한 섭동 RMS는 낮은 음성 크기에서 섭동이 더 두드러질 가능성을 뒷받침한다.

Hann PSD와 원본 크기별 분석에서 약한 구간1~4kHz, 강한 구간1~8kHz와 그 이상 성분을 관찰했다. 전체 FFT의8kHz 이상 점유율은 ε2 약31.4%, 무음 억제형42.8%였지만 절대 고주파 에너지가 증가했다는 뜻은 아니다. ‘바람 소리’를 저주파 단일 원인으로 단정하거나16kHz 이상 모두가 모든 사람에게 들린다고 가정하지 않는다.

근거(ResearchRoot의 로컬 원본): `runs/algorithm_design_20261125/{regional_noise.csv,amplitude_bins.csv,epsilon_saturation.csv,strong_speech_sample_equality.csv,analysis_audit.json}` 및 `runs/segment_noise_balance_20261124/analysis/band_energy.csv`. 구간별 발화 평균과 중첩 프레임 pooled energy를 섞지 않는다. 원래 `ALGORITHM_PLAN_KO.txt`는 `run_pilot.py:94`의 watch()/해시 입력이므로 **기존 경로를 유지**한다.

## 고정 STFT 지각 가중 실험

기존 접근은 원본 STFT power의3×3 평균×0.02+전체 최대 power×1e-6을 budget으로 쓰고,200Hz 미만2배/4kHz 이상4배/8kHz 이상8배 벌점을 줬다. `L_SPEC + scale*L_perceptual`, `scale=strength*mean(abs(g_SPEC))/(mean(abs(g_perceptual))+1e-12)` 뒤 부호 갱신한다. masker/Bark/ATH를 쓰는 실제 역치 모델이 아니며 평균 벌점은 일부 bin 초과를 가릴 수 있다.

| 조건 | 진단1발화 SNR dB | STOI | 신규 공격 SIM, n30 | macro CER, n30 |
|---|---:|---:|---:|---:|
| ε2/10 기준 | 23.60770 | 0.980220 | 0.427286 | 37.8625% |
| ε1/5 | 35.09352 | 0.999138 | 0.503501 | 20.4393% |
| ε2/10+가중2 | 30.65634 | 0.997579 | 0.514290 | 18.8233% |
| ε2/10+가중0.5 | 26.26782 | 0.989078 | 0.465062 | 23.7939% |

출처: `runs/perceptual_spec_20261124/joint_comparison.csv`, `results.json`, `pilot_split_audit.json`과 각 `attacks*/`의 설정/로그. 기준은 보존 모델이고 세 후보는 별도 train140·seed52·1000-step이다. 위 품질은 단일 진단 입력이며140개 평균이 아니다. ε1/1은 최소 섭동 후보였지만 공격 미검증이다.

저장된 iter9 갱신 전 목적 감소율50.32%→48.17%(가중0.5)→41.87%(가중2)와 공격 SIM은 다른 측정이다. 기울기 충돌은 가능한 설명이나 실제 내적 로그가 없어 인과로 확정하지 않는다. 진단1개 test 겹침·후속 후보 사후선택·같은 평가문장 재사용 한계와 실제 청취 미완료를 유지한다.

## 원본 기반 청각 마스킹 역치

외부 패키지를 설치/복사하지 않고 Qin2019식 동시 마스킹을 근사 구현했다. 참고 수식 출처는 원래 `METHOD_KO.txt`에 보존하며 실제 재생 음압 역치를 측정한 것으로 주장하지 않는다.

- 원본 PCM16/44.1kHz/mono에서 periodic Hann STFT2048/512와1024/256, center=True/zero padding을 사용한다. PSD는 `abs(STFT/N)^2`다.
- 원본 한 발화 전체 PSD peak를96dB로 두는 공통 offset을 섭동에도 그대로 적용한다. 섭동을 따로 재정규화하거나 보호본으로 역치를 올리지 않는다.96dB는 실제 SPL 측정값이 아니다.
- ATH 이상인20Hz~20kHz 국소 최대점의 인접3bin power를 합하고0.5Bark 안에서는 강한 masker를 선택한다. noise masker를 별도 분류하지 않고 tonal로 취급하는 근사다.
- `Bark=13atan(0.00076f)+3.5atan((f/7500)^2)`. ATH는 `3.64u^-0.8 - 6.5exp(-0.6(u-3.3)^2)+0.001u^4`, `u=f/1000`이다.
- masker offset은 `-6.025-0.275Bark`; 하방27·ΔBark, 상방 `[-27+0.37max(level-40,0)]·ΔBark`의 두 기울기로 확산한다. 각 masker 역치와 ATH를 선형 power로 합한다.
- margin3/6dB는 역치 power에 `10^(-margin/10)`을 곱한다. **6dB가 더 엄격하다.** ATH 범위 밖은 추가−100dBFS/STFT-bin 공학 상한을 두며 생리적 역치로 해석하지 않는다.
- 시간적 전후 마스킹·개인 청력·헤드폰/스피커 응답·SPL·전달경로·codec은 모델링하지 않았다.

SPEC 갱신 뒤 원본 활동 envelope의 시간별 ε box와 두 STFT magnitude 제약을 교대 투영한다. 내부1dB reserve와 최종 PCM grid 보정을 사용하고, 필요했던 잔차 감쇠 fallback을 기록한다. feasible PCM 중 목적이 개선된 것만 수락한다. 원음 전체 볼륨을 바꾸지 않았지만 잔차가 작아진 비용을 숨기지 않는다. 고정 횟수 투영의 수렴을 가정하지 않고 최종 PCM을 독립 NumPy FFT로 다시 검사한다. 전력비1e-6(약0.0000044dB) 수치 허용 외에 목표 역치를 완화하지 않는다.

## 같은5발화 파일럿 결과

KSS train ID는06460/07448/00347/11187/01638, seed52/beta10/ε2/10회다. 목적 점수는 final PCM에 같은 forward/Gaussian RNG 관례를 적용해 재측정했다. 과거 iter9 로그나 기존 생성의 iteration 간 RNG와 혼용하지 않는다.

| 조건 | SNR dB | STOI | 무음 RMS | 약한 발화 RMS | 강한 발화 RMS | ε2 대비 목적 개선량 보존 |
|---|---:|---:|---:|---:|---:|---:|
| ε2/10 | 23.11649 | 0.98372114 | 0.00668546 | 0.00528610 | 0.00484354 | 100% |
| silence_only | 26.23056 | 0.98372114 | 0.00265389 | 0.00528602 | 0.00484354 | 별도 원본 참조 |
| mask3dB | 41.08362 | 0.99983394 | 0.00045968 | 0.00100831 | 0.00099315 | 10.09835% |
| mask6dB | 53.27944 | 0.99995742 | 0.00022844 | 0.00049914 | 0.00050201 | 5.37393% |

출처: `runs/psychoacoustic_spec_pilot_20261125/pilot/{summary.csv,quality_per_file.csv,masking_per_file.csv,objective_per_file.csv,REPORT_KO.txt}`. 신규 10개 WAV 모두 최종 clipping 0, strict ε 통과, 자기 margin의 두 해상도 위반 0bin/최대 0dB, 전체 Clean fallback 0이었다. 발화 중 잔차 수치는 줄었지만 실제 사람 청취나 신규 공격 SIM/CER는 없다.

보존율은 발화별 `(L_clean-L_candidate)/(L_clean-L_eps2)`의 평균이며 작은 L이 강한 생성 목적이다. 실제 복제 방어율이 아니다. silence_only의 proxy 손실이 ε2보다 약해도 실제 SIM은 더 낮았으므로 단조 관계를 가정하지 않는다. 탐색적95% guard도 방어 성공 임계값이 아니며3/6dB 모두0/5 통과였다.

## 3dB 제약 투영 개선 A·B

동일5개·3dB·ε2·10회·seed52와 같은 두 해상도/시간 box를 유지했다. A(anchor_backtrack)는 위반 제안을 현재 feasible PCM 쪽으로 줄이고, B(local_caps_anchor)는 위반 bin을 최대8pass 국소 보정한 뒤 같은 anchor를 쓴다. 각각 고정 설정1회씩만 비교했다.

| 조건 | SNR dB | STOI | 목적 개선량 보존 | 최종 마스킹/ε 위반 |
|---|---:|---:|---:|---|
| 기존3dB | 41.08362 | 0.99983394 | 10.09835% | 0/0 |
| A | 52.46778 | 0.99998138 | 3.29279% | 0/0 |
| B | 52.17485 | 0.99997940 | 3.93838% | 0/0 |

출처: `runs/psychoacoustic_spec_refinement_20261126/comparison/{summary.csv,objective_per_file.csv,quality_per_file.csv,masking_per_file.csv}`. 기존 5개와 신규 10개 모두 clipping 0이다. 두 후보는 5/5발화에서 기존 3dB보다 목적이 악화돼 채택하지 않았다.

단계 진단50개 제안에서 첫 연속 투영은 평균 SPEC 손실+875.18, 추가 투영+0.251, 첫 rounding−0.119, reserve 보정+13.21, 최종 gain/PCM+29.15였다. 주요 비용은 첫 투영이며 단순 rounding만의 문제가 아니다. 시간 box·두 STFT·여유의 기여는 분리하지 않았다. 첫 투영 뒤47/50, 추가 뒤45/50, rounding 뒤50/50이 위반했고 최종 보정 뒤0/50이었다.

첫 진단은 과거 PCM 일치 검사에서 중단됐고, 재실행5개는 과거 WAV와0/5 일치했다. 근본 원인은 미확정이다. 진단 재실행 평균 보존9.44646%와 고정된 과거 WAV의10.09835%를 혼용하지 않는다. 원문·failure JSON·초기 코드·단계 CSV는 유지한다. CPU 단위검사3개 PASS는 GPU 바이트 재현의 증명이 아니다.

추가 설정 탐색을 반복하지 않고 종료했다.140개 확대·Melo1000-step·SIM/CER·앱 연동은 실행하지 않았다. 다음 연구도 현3dB를 방어 성공 설정으로 취급하지 않고 후보 고정→최종 PCM 제약→사람 청취→독립된 동일 조건 공격 순서로 확인한다.

## 보존 경로와 후속 가설

원래 설계 `runs/algorithm_design_20261125/ALGORITHM_PLAN_KO.txt`, 구현 명세 `runs/psychoacoustic_spec_pilot_20261125/code_worktree/scripts/psychoacoustic_spec_pilot/{METHOD_KO.txt,REFINEMENT_METHOD_KO.txt}`와 실험1차 보고서는 그대로 둔다. 해당 소스9개는 아직 미커밋 상태인 별도 마스킹 worktree에 있다. 이 문서는 구현 파일이나 입력 해시의 대체물이 아니다.

화자 임베딩/학습시점 목적의 보완은 설계 가설로 남아 있다. 기존 평가 encoder만으로 방어를 확정하거나 신규 데이터 학습 없이 기존 모델 지표를 대입하지 않는다. 과거 GPU 시간·저장 공간 예산은 계획 추정이며 새 실행 결과가 아니다. 현재 사람 품질과 강한 복제 방어를 함께 만족한 실사용 설정은 확인되지 않았다.
