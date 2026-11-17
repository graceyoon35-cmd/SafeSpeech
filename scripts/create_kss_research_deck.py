"""Korean slide draft from verified real-KSS measurements, no synthetic mixing."""
import argparse
import json
from pathlib import Path
import zipfile

from pptx import Presentation
from pptx.util import Inches,Pt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifacts',type=Path,required=True);args=p.parse_args()
    data=json.loads((args.artifacts/'kss_results.json').read_text())
    if data['status']!='KSS_REAL_RECORDING_RESEARCH_RESULTS_VERIFIED':raise ValueError('Verified KSS results required')
    prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)

    def text(slide,value,x,y,w,h,size=20):
        box=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));frame=box.text_frame;frame.word_wrap=True
        for i,line in enumerate(value.split('\n')):
            par=frame.paragraphs[0] if i==0 else frame.add_paragraph();par.text=line
            par.font.name='맑은 고딕';par.font.size=Pt(size);par.space_after=Pt(10)

    def slide(title):
        s=prs.slides.add_slide(prs.slide_layouts[6]);text(s,title,.55,.3,12,.65,28)
        text(s,'KSS 실제 녹음 · 비상업 내부 연구 · 단일 화자/seed · 전수 청취 미실시',.55,7.02,12,.3,10)
        return s

    s=slide('SafeSpeech — KSS 실제 음성 실험 결과')
    text(s,'• KSS 200개 유지: train 140 / validation 30 / test 30\n• clean / SPEC_10 / SPEC_1 각 1,000-step 실제 학습\n• 동일 초기 가중치·대본·특징·seed 52·batch 1·BF16\n• 미청취·ASR 불일치 상태를 보존하고 자동 검증 데이터 활용\n• 기존 합성 100-step 실험과 결과/경로 분리',.85,1.5,11.6,4.5,24)
    s=slide('KSS SPEC 생성과 입력 품질 검증')
    s.shapes.add_picture(str(args.artifacts/'kss_spec_stoi.png'),Inches(.7),Inches(1.1),width=Inches(7.4))
    spec=data['spec']
    text(s,f"400개 보호 WAV\n2,200회 gradient 검증\n평균 STOI\nSPEC_1: {spec['stoi_mean_spec1']:.4f}\nSPEC_10: {spec['stoi_mean_spec10']:.4f}\nepsilon = 8/255",8.4,1.5,4.1,4.5,22)
    text(s,'STOI는 입력 명료도 지표입니다. 주관적 청취 품질이나 복제 방어 성공 자체가 아닙니다.',.7,6.2,12,.5,15)
    s=slide('동일 조건 1,000-step 실제 학습')
    s.shapes.add_picture(str(args.artifacts/'kss_training.png'),Inches(.55),Inches(1.3),width=Inches(12.2))
    text(s,'공식 Melo 생성기·손실 구성요소를 사용하는 단일 GPU 제한 학습 루프. 판별기는 같은 seed로 초기화했습니다.',.65,6.25,12,.55,15)
    s=slide('고정 test 30문장 SIM·CER')
    s.shapes.add_picture(str(args.artifacts/'kss_sim_cer.png'),Inches(1.0),Inches(1.0),width=Inches(11.3))
    text(s,'SIM: 같은 clean train 기준 녹음 5개 / CER: Whisper small 한국어. 구간은 문장 bootstrap만 반영합니다.',.65,6.3,12,.5,14)
    s=slide('실측 수치와 해석 범위')
    rows=[]
    for r in data['evaluation']:
        rows.append(f"{r['condition']}: SIM {r['mean_sim']:.4f}, 문장 평균 CER {r['mean_cer']*100:.2f}%")
    text(s,'\n'.join(rows),.85,1.25,11.6,2.5,24)
    text(s,'• CER는 삽입 오류로 100%를 넘을 수 있음 (과다 전사 영향 포함)\n• 단일 화자/seed·미청취 대본, 문장별·decoder 민감도 별도 보존\n• 사전학습 코퍼스와 KSS의 비중복 미확인\n• 다중 공격 모델·채널 변환 후 재학습 방어 미검증\n• 합성 100-step과 KSS 결과를 합산하지 않음',.85,3.85,11.6,2.85,18)
    s=slide('출처·검수 상태·재현 기록')
    text(s,'• Kyubyong Park, KSS Dataset, 2018 / CC BY-NC-SA 4.0\n• 원 배포: kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset\n• 실제 미러: Bingsu/KSS_Dataset, 고정 리비전 48fdfd7\n• 공개 조건의 비상업 TTS 연구 범위로 로컬 실험 수행\n• 별도 화자 동의 미확인, 음성/가중치 외부 배포 미실행\n• ASR 불일치 58개 검수 권장, 학습 제외 0개\n• 원본·split·WAV·checkpoint 해시와 step 로그 보존\n• 상세 데이터: kss_results.json, kss_training.csv, kss_per_file.csv',.85,1.2,11.7,5.6,20)
    target=args.artifacts/'SafeSpeech_KSS_research_20261118.pptx'
    if target.exists():raise FileExistsError('Existing slide draft preserved')
    prs.save(target);checked=Presentation(target)
    for s in checked.slides:
        for shape in s.shapes:
            if shape.left+shape.width>checked.slide_width or shape.top+shape.height>checked.slide_height:
                raise ValueError('Shape exceeds slide bounds')
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:raise ValueError('PPTX ZIP invalid')
    print(json.dumps({'pptx':str(target.resolve()),'slides':len(checked.slides),'validation':'Package/bounds only; Office rendering not tested'}))


if __name__=='__main__':main()
