"""Create a Korean PPTX draft from verified benchmark artifacts only."""
import argparse
import json
from pathlib import Path
import zipfile

from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',type=Path,required=True)
    args=p.parse_args()
    report=json.loads((args.artifacts/'final_report.json').read_text())
    if report['status']!='VERIFIED_SYNTHETIC_100_STEP_BENCHMARK':raise ValueError('Verified data required')
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
    color=RGBColor(25,48,75);muted=RGBColor(85,95,110)

    def text(slide,value,x,y,w,h,size=20,bold=False):
        box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));frame=box.text_frame;frame.word_wrap=True
        for i,line in enumerate(value.split('\n')):
            par=frame.paragraphs[0] if i==0 else frame.add_paragraph();par.text=line
            par.font.name='맑은 고딕';par.font.size=Pt(size);par.font.bold=bold;par.font.color.rgb=color
            par.space_after=Pt(10)
        return box

    def slide(title):
        s=prs.slides.add_slide(prs.slide_layouts[6]);text(s,title,.55,.3,12.2,.65,28,True)
        box=text(s,'합성 데이터 검증 | KSS 실제 음성 SPEC·학습 미실행 | 2026-11-18',.55,7.02,11.6,.3,10)
        box.text_frame.paragraphs[0].font.color.rgb=muted
        text(s,str(len(prs.slides)),12.35,7.02,.4,.3,10)
        return s

    s=slide('SafeSpeech — 데스크톱 통합 실험 현황')
    text(s,'100-step 학습·복구·평가 경로 검증 완료',.75,1.25,11.7,.8,29,True)
    text(s,'• clean / SPEC_1 / SPEC_10: 각 100-step 성공\n• 50-step 저장 후 새 프로세스에서 optimizer·난수 상태 복구\n• 고정 12문장 SIM·CER 평가 및 문장별 통계 생성\n• 기존 3-step 결과·체크포인트 보존\n• KSS 200개 준비 완료, 사용 확인·청취 검수는 대기',.85,2.3,11.7,3.8,23)
    text(s,'이 자료는 합성 데이터 2발화·단일 학습 seed의 기술 실험입니다.\nKSS 화자 보호 효과나 일반화된 방어 성능을 입증하지 않습니다.',.85,6.05,11.7,.8,16,True)
    s=slide('실제 실행 환경과 동일 조건')
    text(s,'• Windows + WSL2 Ubuntu 22.04 / RTX 5060 Ti 8GB\n• Python 3.10.12 / PyTorch·Torchaudio 2.7.1+cu128\n• batch 1 / segment 8192 / BF16 / AdamW / LR 0.0001\n• 학습 seed 52, 동일 사전학습 생성기 가중치\n• 동일 대본·split·BERT 특징 / train 2, validation 1\n• 공식 Melo 모델·손실을 사용한 단일 GPU 제한 학습 루프',.8,1.3,11.7,4.9,23)
    text(s,'공식 train.py 전체 실행과 구분합니다. 판별기는 같은 seed의 무작위 초기화입니다.',.8,6.25,11.7,.5,16)
    s=slide('100-step 손실·시간·GPU 메모리 실측')
    s.shapes.add_picture(str(args.artifacts/'training_100step.png'),Inches(.55),Inches(1.15),width=Inches(12.2))
    lines=[' / '.join(f"{r['condition']}: {r['mean_step_seconds']:.3f}초/step, {r['peak_allocated_gib']:.3f}GiB" for r in report['training'])]
    text(s,lines[0],.65,5.75,12,.65,16)
    text(s,'step 시간은 CUDA 동기화 후 측정했으며 체크포인트 저장을 제외합니다.\n손실의 감소나 조건 간 차이만으로 방어 성공을 판단하지 않습니다.',.65,6.25,12,.65,14)
    s=slide('SIM·CER — 고정 12문장 실제 측정')
    s.shapes.add_picture(str(args.artifacts/'evaluation_12prompts.png'),Inches(1.0),Inches(1.05),width=Inches(11.3))
    text(s,'오차 막대: 문장 재표집 bootstrap 95% 구간. 화자·학습 seed 간 불확실성은 포함하지 않습니다.',.65,6.35,12,.5,14)
    s=slide('SPEC 및 강건성 채널 변환 검증')
    s.shapes.add_picture(str(args.artifacts/'robustness_quality.png'),Inches(.6),Inches(1.05),width=Inches(12.1))
    text(s,'실제 합성 test 입력 1개 × SPEC 1/10회 × 15조건 × 2회 반복\nWAV 120개·MP3 24개 검증. 같은 seed의 반복은 독립 표본이 아닙니다.\nSTOI는 변환 후 품질 지표이며, 변환 데이터 재학습에 대한 방어 성능은 미측정입니다.',.7,5.65,11.9,1.15,15)
    s=slide('본 학습 시간 추정 — KSS 실측값 아님')
    rows=[]
    for steps in (1000,5000,10000):
        a=next(r for r in report['estimates'] if r['steps_per_condition']==steps and r['conditions']=='clean+SPEC_10')
        b=next(r for r in report['estimates'] if r['steps_per_condition']==steps and r['conditions']=='clean+SPEC_1+SPEC_10')
        rows.append(f"각 {steps:,} step: clean+SPEC_10 약 {a['estimated_training_and_save_minutes']:.0f}분 / 세 조건 약 {b['estimated_training_and_save_minutes']:.0f}분")
    text(s,'\n'.join(rows),.85,1.4,11.7,2.3,24,True)
    storage=report['storage']
    text(s,f"• WSL 내부 저장 실측 {storage['native_checkpoint_save_seconds']:.2f}초 (Windows 경로 {storage['windows_checkpoint_save_seconds']:.2f}초)\n• 위 추정은 500-step 저장 간격과 합성 데이터 실측 속도를 적용\n• 일정에는 50% 여유를 별도 반영 권장\n• 사용 확인·검수·SPEC·전처리·평가 시간은 제외\n• 실제 KSS pilot 결과에 따라 추정 보정 필요",.85,3.8,11.7,2.7,20)
    s=slide('KSS 검수 및 승인 후 실행 준비')
    text(s,'• 200 WAV / 고정 train 140·validation 30·test 30 보존\n• 대본·해시·split 검증 및 한국어 BERT 특징 준비\n• 청취 검수 0/200, 사용 확인 플래그 0/200 그대로 유지\n• Review-KSS.cmd: 재생·대본 대조·판정·메모·자동 저장\n• 검수 CSV + 실제 팀 사용 범위 기록 → 별도 승인 입력 생성\n• 승인 입력 → SPEC → 동일 조건 학습 → test 평가 자동 연결',.8,1.3,11.7,4.8,22)
    text(s,'CC BY-NC-SA 4.0의 비상업 TTS 근거와 별도 미확인 권리를 구분합니다.\n합성물·가중치 외부 공개는 자동 승인하지 않았습니다.',.8,6.15,11.7,.7,16)
    s=slide('완료한 기능과 남은 검증')
    text(s,'완료\n• GPU SPEC·100-step paired 학습·주기적 체크포인트·복구\n• 12문장 SIM/CER·발화별 CSV·평균/분산/문장 bootstrap\n• 채널 변환 품질 평가·그래프·로컬 청취 검수 도구\n\n미완료\n• KSS 청취·사용 범위 확인과 실제 KSS SPEC/학습\n• 다중 학습 seed·다중 화자 일반화 검증\n• 채널 변환 후 재학습 공격에 대한 방어 평가',.85,1.15,11.8,5.6,21)
    s=slide('11월 21일 결과 확보를 위한 실행 계획')
    text(s,'11/18  합성 100-step·복구·평가 완료, 검수 도구 전달\n11/19  팀 사용 범위 기록 + 우선 청취 검수, 승인 표본 pilot\n11/20  실제 KSS clean/protected 동일 step 학습·고정 test 평가\n11/21  결과/체크포인트/CSV 고정, 필요 시 학습량·seed 확대\n11/22–23  재검증·PPT 표와 그래프 확정\n11/24  발표 자료 마감',.85,1.4,11.7,4.8,23)
    text(s,'전제: 사용·검수 조건 충족. 지연되면 합성 실험 결과와 미완료 범위를 명확히 발표합니다.',.85,6.25,11.7,.55,15,True)
    s=slide('재현성 및 근거 파일')
    text(s,'• 모델·설정·입력·WAV·체크포인트 SHA-256 보존\n• 기존 3-step 산출물의 저장 당시 해시와 재비교 완료\n• 복구 후 optimizer step 50 → 100 확인\n• 같은 step50 체크포인트의 두 재시작 분기에서 step51/52 손실 일치\n• 자동 테스트 84개 PASS, Windows 검수 UI 구성 검사 PASS\n• 자료: training_timing.csv / evaluation_per_file.csv / final_report.json\n• 모든 수치의 범위: 합성 소규모 실험 또는 조건부 시간 추정',.85,1.25,11.7,5.3,21)
    output=args.artifacts/'SafeSpeech_synthetic_benchmark_20261118.pptx'
    if output.exists():raise FileExistsError('Refuse to overwrite an existing deck')
    prs.save(output)
    # Validate package/readability/bounds, not a claim of an Office rendering test.
    check=Presentation(output)
    for s in check.slides:
        for shape in s.shapes:
            if shape.left<0 or shape.top<0 or shape.left+shape.width>check.slide_width or shape.top+shape.height>check.slide_height:
                raise ValueError('Slide shape exceeds slide bounds')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:raise ValueError('Invalid PPTX package')
    print(json.dumps({'pptx':str(output.resolve()),'slides':len(check.slides),'validation':'PPTX package, text and shape bounds; Office rendering not tested'}))


if __name__=='__main__':main()
