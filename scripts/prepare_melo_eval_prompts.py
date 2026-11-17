"""Fixed novel Korean evaluation prompts, real BERT features and immutable hashes."""
import argparse
import json
from pathlib import Path
import tempfile
import torch

from melo_training_core import KoreanFrontend,training_config
from run_korean_melo_spec import ROOT,save,sha256
from evaluate_paired_melo_smoke import normalize

TEXTS=[
 '이 문장은 평가에만 사용합니다.', '작은 도서관에서 조용히 책을 읽었습니다.',
 '내일 아침에는 공원에서 산책할 예정입니다.', '창문을 열자 시원한 바람이 들어왔습니다.',
 '친구와 함께 맛있는 저녁을 준비했습니다.', '새로운 기술을 배우는 일은 즐겁습니다.',
 '기차가 도착하기 전에 표를 확인해 주세요.', '비가 그친 뒤 하늘에 무지개가 나타났습니다.',
 '중요한 자료는 안전한 장소에 보관합니다.', '다양한 문장으로 음성 품질을 확인합니다.',
 '따뜻한 차를 마시며 오늘의 계획을 세웁니다.', '연구 결과는 정확한 기록과 함께 설명합니다.'
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--bert',type=Path,required=True)
    args=p.parse_args()
    dataset=json.loads((args.experiment/'clean/dataset.json').read_text())['utterances']
    training_texts={normalize(r['text']) for r in dataset if r['split']!='test'}
    if len(set(map(normalize,TEXTS)))!=len(TEXTS) or training_texts & set(map(normalize,TEXTS)):
        raise ValueError('Duplicate evaluation text or training/validation overlap')
    torch.set_num_threads(4)
    frontend=KoreanFrontend(args.bert,training_config(args.assets))
    folder=Path(tempfile.mkdtemp(prefix='melo_eval_prompts_12_',dir=ROOT/'runs'))
    rows=[]
    for i,text in enumerate(TEXTS):
        uid=f'eval_{i:03d}';path=folder/(uid+'.features.pt')
        torch.save(frontend.features(text),path)
        rows.append({'utterance_id':uid,'text':text,'features':str(path),'features_sha256':sha256(path)})
    save(folder/'evaluation_prompts.json',{'schema_version':1,'base_seed':2026,'prompts':rows,
        'training_experiment':str(args.experiment.resolve()),'scope':'synthetic_benchmark_only',
        'training_validation_text_overlap':False})
    print(json.dumps({'manifest':str(folder/'evaluation_prompts.json'),'count':len(rows)}))


if __name__=='__main__':
    main()
