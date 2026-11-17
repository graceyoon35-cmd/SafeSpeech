"""Generate nonhuman Korean speech and preserve fixed splits for training smoke."""
import argparse
from pathlib import Path
import tempfile
import torch
import soundfile as sf

from melo_training_core import KoreanFrontend, infer, model_from_config, training_config
from run_korean_melo_spec import ROOT, save, sha256, MODEL, OFFICIAL


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--bert', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    cfg = training_config(args.assets)
    frontend = KoreanFrontend(args.bert, cfg)
    model = model_from_config(cfg, args.assets / 'model/checkpoint.pth').eval()
    folder = Path(tempfile.mkdtemp(prefix='synthetic_melo_training_', dir=ROOT/'runs'))
    texts = [('train', '안녕하세요. 음성 보호를 연구합니다.'),
             ('train', '오늘은 맑고 따뜻한 날입니다.'),
             ('validation', '한국어 음성을 안전하게 지킵니다.'),
             ('test', '이 문장은 평가에만 사용합니다.')]
    rows = []
    for i, (split, text) in enumerate(texts):
        uid = f'synth_{i:03d}'
        feature = frontend.features(text)
        wave = infer(model, feature, seed=52+i)
        path = folder/f'{uid}.wav'
        sf.write(path, wave, 44100, subtype='PCM_16')
        if not 1 <= len(wave)/44100 <= 10:
            raise ValueError('Synthetic utterance outside runner length limits')
        torch.save(feature, folder/f'{uid}.features.pt')
        rows.append(dict(utterance_id=uid, input_kind='model_synthesized', language='ko',
            speaker_id='synthetic_KR', model_speaker='KR', split=split, text=text,
            transcript_verified=True, transcript_reference='Generation prompt only; not listening reviewed.',
            original_path=path.name, original_sha256=sha256(path), sample_rate=44100,
            source=f'MeloTTS-Korean public pretrained model {MODEL}; seed {52+i}',
            research_use_permitted=True, permission_scope='synthetic_technical_test',
            permission_reference='Public model card MIT; local synthetic pipeline test, no new human target.',
            synthetic_evidence_path='generation.json', features_path=f'{uid}.features.pt'))
        print(f'{uid}: {len(wave)/44100:.3f}s', flush=True)
    save(folder/'generation.json', {'input_kind': 'model_synthesized', 'model_revision': MODEL,
        'source_revision': OFFICIAL, 'speaker': 'KR', 'human_recording': False,
        'listening_review': 'NOT_PERFORMED',
        'limitations': 'Prompt provenance only; no human consent or defense-efficacy claim.',
        'generated_texts': [r['text'] for r in rows]})
    # Existing runner checks generation text provenance for nonhuman inputs only
    # through declarations. No KSS review or permission field is changed.
    save(folder/'input.json', {'schema_version': 1, 'utterances': rows})
    save(folder/'base_config.json', cfg)
    print(str(folder), flush=True)


if __name__ == '__main__':
    main()
