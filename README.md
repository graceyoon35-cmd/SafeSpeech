# SafeSpeech — 한국어 SPEC 연구·시연

원논문 SafeSpeech(USENIX Security 2025)를 바탕으로 Windows/WSL·MeloTTS에서 복제 방어와 저잡음 처리를 연구한다. 아래는 이 프로젝트의 **2026-11-26까지 보존된 실험 기록**이며 원논문 전체 재현이나 제품 성능 보장이 아니다.

## 개발자가 먼저 볼 문서

| 목적 | 공식 문서 |
|---|---|
| 검증된 결과·조건·미검증 범위 | [최종 보고서](docs/SPEC_FINAL_REPORT_20261120_KO.md) |
| 실행·시연·청취 평가·재개 | [실행 및 인계](docs/README_SPEC_HANDOFF_20261120_KO.md) |
| 알고리즘·잡음·지각 가중·마스킹·투영 | [SPEC 알고리즘](docs/SPEC_ALGORITHM_KO.md) |
| 앱 입출력·비동기·저장·팀 반영 범위 | [앱 연동 계약](docs/SPEC_APP_INTEGRATION_REVIEW_20261120_KO.md) |

기존 날짜가 붙은 공식 문서명을 유지한다. 같은 역할의 `SPEC_FINAL_REPORT_KO.md`·`SPEC_HANDOFF_KO.md`를 별도로 만들지 않는다. 과거 보고서·실험 원문·CSV/JSON·manifest·비공개 정답표는 그대로 보존한다.

## 현재 판단

- 기존 ε=8/255·10회는 제한된 Melo 복제 공격에서 방어 방향이 관찰됐지만, 보호 입력 평균 SNR 약11.79dB/STOI 약0.902로 잡음 비용이 크다.
- ε=2/255·10회는 기존 시연 프로필이다. silence_only_smooth는 train140 평균 SNR26.22dB/STOI0.98117, test30 SIM0.3903을 보였으나 발화 중 잡음·기존 방어 대비 비열등성·실사용 청취 검증은 해결되지 않았다.
- 청각 마스킹 3/6dB와 투영 A/B는 **5발화 품질·제약·생성 목적만 측정**했다. 새 Melo 공격 SIM/CER는 없다. A/B는 목적 보존이 악화되어 채택하지 않았다.
- 불가청·복제 완전 차단·확률형 신뢰도를 주장하지 않는다. 청취 패키지 준비는 사람 평가 완료와 다르다.

## 실행 경로

| 경로 | 역할 |
|---|---|
| 이 연구 작업공간 `SafeSpeech-spec-perturbation` | 최신 시연 프로필·호출 코드·저잡음 실험·문서 |
| 별도 `SafeSpeech-desktop-integration-20261118` | 검증된 `protect_wav.py`·Windows/WSL 브리지·Melo 학습/평가 backend |
| `/home/safespeech/.venvs/melo-training-20261118` | 기존 학습·보호 Python 환경 |
| `/home/safespeech/models/melo-spec-pinned-20261118` | 고정 모델·설정·소스 자산 |

현재 문서 통합 브랜치는 backend 코드를 병합하지 않는다. 이 작업공간에서 존재하지 않는 `scripts/protect_wav.py`를 실행하지 말고 인계 문서의 backend 기준 명령을 따른다. 기존 환경을 재사용하며 원논문 `requirements.txt`를 현재 Melo 환경에 덮어 설치하지 않는다.

CLI `--epsilon 2`는 정규화 진폭 **2/255**다. `--epsilon 2/255` 문자열을 전달하지 않는다. 앱은 새 UUID 작업의 검증된 `protected.wav`만 재생·저장한다. 자세한 실제 명령과 입력 제한은 [인계](docs/README_SPEC_HANDOFF_20261120_KO.md)에 있다.

## 연구 기록과 재현

품질은 보호 입력의 SNR/STOI, 방어는 보호 데이터로 새로 학습한 공격 모델 합성의 SIM/CER로 구분한다. 서로 다른 표본·seed·저장 보정·모델의 값을 한 실험처럼 합치지 않는다. `runs/`·데이터·체크포인트·백업은 Git 제외인 로컬 증거이며, 새 clone에는 포함되지 않는다. 경로와 지표 출처는 공식 문서에 명시했다.

새 학습·보호 생성·앱 시연은 이 문서 통합 작업에서 실행하지 않았다. 기존 GUI와 테스트 통과 기록은 해당 날짜의 결과이며 오늘 다시 검증한 것으로 표시하지 않는다.

## 원논문과 출처

[SafeSpeech 논문](https://arxiv.org/pdf/2504.09839) · [원논문 데모](https://wxzyd123.github.io/safespeech) · [BERT-VITS2](https://github.com/fishaudio/Bert-VITS2)

KSS 출처·변경 표시·사용 조건과 확인 한계는 [최종 보고서의 보존 원문](docs/SPEC_FINAL_REPORT_20261120_KO.md)에 있다. 별도 화자 동의나 음성·가중치 외부 공개 승인을 문서 통합으로 새로 확보한 것은 아니다.

## Citation

If you find our repository helpful, please consider citing our work in your research or project.

```
@inproceedings{zhang2025safespeech,
  author = {Zhang, Zhisheng and Wang, Derui and Yang, Qianyi and Huang, Pengyang and Pu, Junhan and Cao, Yuxin and Ye, Kai and Hao, Jie and Yang, Yixian},
  title = {SafeSpeech: Robust and Universal Voice Protection Against Malicious Speech Synthesis},
  booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
  year = {2025},
  address = {Seattle, WA, USA}
}
```
