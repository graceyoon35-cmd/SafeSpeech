"""Pinned Melo model and Korean-only real frontend; no multilingual import stubs."""
import copy
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from run_korean_melo_spec import check_assets


def setup(assets):
    assets, _, _ = check_assets(assets)
    sys.path.insert(0, str(assets / 'MeloTTS'))
    return assets


def training_config(assets):
    assets = setup(assets)
    cfg = json.loads((assets / 'MeloTTS/melo/configs/config.json').read_text())
    raw = json.loads((assets / 'model/config.json').read_text())
    # Keep the entire pretrained architecture and embedding sizes, including
    # n_speakers=256. Relabel a target speaker to the existing KR slot 0.
    cfg['model'] = copy.deepcopy(raw['model'])
    cfg['data'].update(raw['data'])
    for key in ('symbols', 'num_tones', 'num_languages'):
        cfg[key] = raw[key]
    cfg['train'].update(batch_size=1, segment_size=8192, fp16_run=True,
                        seed=52, learning_rate=0.0001)
    return cfg


def model_from_config(cfg, checkpoint, device='cuda'):
    from melo.models import SynthesizerTrn
    model = SynthesizerTrn(len(cfg['symbols']), cfg['data']['filter_length']//2+1,
        cfg['train']['segment_size']//cfg['data']['hop_length'],
        n_speakers=cfg['data']['n_speakers'], num_tones=cfg['num_tones'],
        num_languages=cfg['num_languages'], **cfg['model']).to(device)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    return model


class KoreanFrontend:
    def __init__(self, bert_path, cfg):
        from transformers import AutoTokenizer
        self.bert_path = str(Path(bert_path).resolve())
        self.cfg = cfg
        original = AutoTokenizer.from_pretrained

        def local(name, *args, **kwargs):
            if name == 'kykim/bert-kor-base':
                name = self.bert_path
            return original(name, *args, **kwargs)

        # Official korean.py initializes its tokenizer at import time.
        with patch.object(AutoTokenizer, 'from_pretrained', side_effect=local):
            self.kr = importlib.import_module('melo.text.korean')
        self.jbert = importlib.import_module('melo.text.japanese_bert')

    def features(self, text):
        from melo.text import cleaned_text_to_sequence
        from melo.commons import intersperse
        norm = self.kr.text_normalize(text)
        phones, tones, counts = self.kr.g2p(norm)
        p, t, lang = cleaned_text_to_sequence(phones, tones, 'KR',
            {s: i for i, s in enumerate(self.cfg['symbols'])})
        expanded = list(counts)
        if self.cfg['data']['add_blank']:
            p, t, lang = (intersperse(items, 0) for items in (p, t, lang))
            expanded = [c*2 for c in expanded]
            expanded[0] += 1
        # Official Korean implementation delegates to this exact feature extractor.
        bert = self.jbert.get_bert_feature(norm, expanded, device='cpu', model_id=self.bert_path)
        if bert.shape != (768, len(p)) or not torch.isfinite(bert).all():
            raise ValueError('Invalid real Korean BERT feature')
        return {'text': text, 'normalized': norm, 'phones': phones, 'tones': tones,
                'word2ph': counts, 'phone_ids': torch.tensor(p, dtype=torch.long),
                'tone_ids': torch.tensor(t, dtype=torch.long),
                'language_ids': torch.tensor(lang, dtype=torch.long), 'ja_bert': bert}


def infer(model, feature, seed=52, device='cuda'):
    torch.manual_seed(seed)
    n = len(feature['phone_ids'])
    with torch.no_grad():
        result = model.infer(feature['phone_ids'][None].to(device),
            torch.tensor([n], device=device), torch.tensor([0], device=device),
            feature['tone_ids'][None].to(device), feature['language_ids'][None].to(device),
            torch.zeros(1, 1024, n, device=device), feature['ja_bert'][None].to(device),
            sdp_ratio=.2, noise_scale=.6, noise_scale_w=.8, length_scale=1.)[0][0, 0]
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite synthesis')
    return result.cpu().numpy()
