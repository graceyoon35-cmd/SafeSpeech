"""Save public source evidence without changing consent or review declarations."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'runs/training_20261118/license_evidence'
out.mkdir(parents=True, exist_ok=True)
urls = {
    'kaggle_view.json': 'https://www.kaggle.com/api/v1/datasets/view/bryanpark/korean-single-speaker-speech-dataset',
    'kaggle_metadata.json': 'https://www.kaggle.com/api/v1/datasets/metadata/bryanpark/korean-single-speaker-speech-dataset',
    'kss_mirror_card.md': 'https://huggingface.co/datasets/Bingsu/KSS_Dataset/resolve/48fdfd7ab1dbc1a62e4e8a8b9f4c360259d51d3c/README.md',
    'cc_by_nc_sa_4_legalcode.html': 'https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en',
    'melo_model_card.md': 'https://huggingface.co/myshell-ai/MeloTTS-Korean/resolve/0207e5adfc90129a51b6b03d89be6d84360ed323/README.md',
    'bert_model_card.md': 'https://huggingface.co/kykim/bert-kor-base/resolve/1779cc0982ada0216dd6de0dd4e86fb78201926d/README.md',
}
records = []
for name, url in urls.items():
    record = {'url': url, 'file': name}
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
            record.update(status=response.status, content_type=response.headers.get('Content-Type'))
        with (out / name).open('xb') as stream:
            stream.write(content)
        record['sha256'] = hashlib.sha256(content).hexdigest()
        if name.startswith('kaggle') and 'json' in record['content_type']:
            obj = json.loads(content)
            record['license_fields'] = {k: v for k, v in obj.items() if 'licens' in k.lower()}
    except Exception as error:
        record['error_type'] = type(error).__name__
        record['error'] = str(error)
    records.append(record)
with (out / 'sources.json').open('x', encoding='utf-8') as stream:
    json.dump(records, stream, ensure_ascii=False, indent=2)
print(json.dumps(records, ensure_ascii=False, indent=2))
