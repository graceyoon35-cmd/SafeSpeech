"""Read-only KSS provenance, saved-WAV quality and independent CER audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

import numpy as np
import soundfile as sf
from pystoi import stoi


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def table(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(text):
    return re.sub(r'[^0-9A-Za-z가-힣]', '', text)


def distance(a, b):
    """Independent unit-cost Levenshtein; no jiwer call."""
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def quality(clean, protected, epsilon):
    a, sr = sf.read(clean, dtype='float64')
    b, br = sf.read(protected, dtype='float64')
    if a.ndim != 1 or a.shape != b.shape or sr != br or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('WAV shape, rate or finite-value mismatch')
    delta = b-a
    power = float(np.sum(delta**2))
    snr = float(10*np.log10(np.sum(a**2)/power)) if power else None
    maximum = float(np.max(np.abs(delta)))
    return {'sample_rate': sr, 'samples': len(a), 'linf': maximum, 'epsilon': epsilon,
            'pcm16_tolerance': 1/32768, 'strict_epsilon_pass': maximum <= epsilon,
            'epsilon_pass': maximum <= epsilon+1/32768+1e-7,
            'snr_db': snr, 'stoi': float(stoi(a, b, sr, extended=False)),
            'clean_peak': float(np.max(np.abs(a))), 'protected_peak': float(np.max(np.abs(b))),
            'clean_rail_samples': int(np.count_nonzero((a <= -1) | (a >= 32767/32768))),
            'protected_rail_samples': int(np.count_nonzero((b <= -1) | (b >= 32767/32768))),
            'out_of_range_samples': int(np.count_nonzero((b < -1) | (b > 32767/32768))),
            'protected_subtype': sf.info(protected).subtype}


def audit(study_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    errors = []
    def check(ok, label):
        if not ok:
            errors.append(label)
    study = load(study_path)
    experiment = Path(study['experiment'])
    trained = load(study['training_summary'])
    evaluation_path = Path(study['evaluation'])
    evaluation = load(evaluation_path)
    evidence = {}
    def verify(path, expected=None):
        path = str(path)
        actual = evidence.setdefault(path, digest(path))
        if expected:
            check(actual == expected, 'hash mismatch: '+path)
        return actual
    configs = []
    signatures = []
    training = []
    datasets = {}
    for row in trained['training']:
        cfgpath = Path(row['run'])/'config.json'
        cfg = load(cfgpath)
        verify(cfgpath)
        configs.append(cfg)
        manifest = cfg['experiment']['dataset_manifest']
        dataset = load(manifest)['utterances']
        verify(manifest)
        datasets[row['condition']] = dataset
        signatures.append([(r['utterance_id'], r['split'], r['text'], r['features_sha256']) for r in dataset])
        report = load(row['report'])
        verify(row['report'])
        verify(row['checkpoint'], report['checkpoint_sha256'])
        verify(cfg['experiment']['generator_initialization'], report['initial_checkpoint_sha256'])
        check(report['completed_steps'] == 1000, 'steps: '+row['condition'])
        check(report['trainer_sha256'] == digest(Path(__file__).with_name('train_melo_bounded.py')), 'trainer changed')
        for item in dataset:
            for key in ('wav', 'features'):
                if item[key] not in evidence:
                    verify(item[key], item[key+'_sha256'])
        training.append({'condition': row['condition'], 'seed': cfg['train']['seed'],
                         'steps': report['completed_steps'], 'checkpoint': row['checkpoint'],
                         'checkpoint_sha256': report['checkpoint_sha256'], 'config': str(cfgpath),
                         'initial_checkpoint_sha256': report['initial_checkpoint_sha256'],
                         'train': cfg['train'], 'data_order': 'cyclic fixed manifest order; no shuffle',
                         'precision': report['precision']})
    check(all(c['train'] == configs[0]['train'] and c['model'] == configs[0]['model'] for c in configs), 'paired hyperparameters')
    check(all(s == signatures[0] for s in signatures), 'paired dataset identity')
    data = datasets['clean']
    split = Counter(r['split'] for r in data)
    check(dict(split) == {'train': 140, 'validation': 30, 'test': 30}, 'split counts')
    text_splits = {}
    for r in data:
        text_splits.setdefault(normalize(r['text']), set()).add(r['split'])
    check(not any(len(v)>1 for v in text_splits.values()), 'normalized text leakage')
    check(len({r['wav_sha256'] for r in data}) == 200, 'duplicate clean recordings')
    with (evaluation_path.parent/'per_file.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    test = {r['utterance_id']: r for r in data if r['split']=='test'}
    audited = []
    for r in rows:
        verify(r['output'], r['output_sha256'])
        ref, hyp = normalize(r['text']), normalize(r['asr_transcript'])
        edit = distance(ref, hyp)
        logged = sum(int(r[k]) for k in ('substitutions', 'deletions', 'insertions'))
        check(edit == logged and len(ref) == int(r['reference_characters']), 'CER counts: '+r['condition']+'/'+r['sentence_id'])
        check(abs(edit/len(ref)-float(r['cer'])) < 1e-12, 'CER rate')
        check(r['sentence_id'] in test and r['text'] == test[r['sentence_id']]['text'], 'test prompt')
        audited.append({**r, 'independent_edits': edit, 'independent_cer': edit/len(ref),
                        'hypothesis_characters': len(hyp), 'above_100_percent': edit > len(ref)})
    summary = []
    for condition in ('pretrained', 'clean', 'SPEC_1', 'SPEC_10'):
        chosen = [r for r in audited if r['condition']==condition]
        check(len(chosen)==30 and {r['sentence_id'] for r in chosen}==set(test), 'evaluation coverage: '+condition)
        check([int(r['evaluation_seed']) for r in chosen] == list(range(2026,2056)), 'evaluation seed')
        summary.append({'condition': condition, 'n': len(chosen),
            'sim': float(np.mean([float(r['sim']) for r in chosen])),
            'macro_cer': float(np.mean([r['independent_cer'] for r in chosen])),
            'corpus_cer': sum(r['independent_edits'] for r in chosen)/sum(int(r['reference_characters']) for r in chosen),
            'insertions': sum(int(r['insertions']) for r in chosen),
            'deletions': sum(int(r['deletions']) for r in chosen),
            'substitutions': sum(int(r['substitutions']) for r in chosen),
            'above_100_count': sum(r['above_100_percent'] for r in chosen),
            'maximum_cer': max(r['independent_cer'] for r in chosen)})
    table(output/'cer_per_file.csv', audited)
    table(output/'cer_summary.csv', summary)
    pairs = load(Path(study['spec_run'])/'handoff_manifest.json')['utterances']
    measurements = []
    for i, r in enumerate(pairs):
        for role in ('original', 'clean', 'protected'):
            if r[role+'_path'] not in evidence:
                verify(r[role+'_path'], r[role+'_sha256'])
        check(r['original_sha256']==r['clean_sha256'], 'clean copy identity')
        q = quality(r['clean_path'], r['protected_path'], r['epsilon'])
        check(q['epsilon_pass'] and q['out_of_range_samples']==0 and q['sample_rate']==44100 and q['protected_subtype']=='PCM_16', 'WAV contract: '+r['utterance_id'])
        measurements.append({'condition': 'SPEC_'+str(r['iterations']), 'utterance_id': r['utterance_id'],
                             'split': r['split'], 'original': r['original_path'], 'protected': r['protected_path'],
                             **q, 'previous_stoi': r['stoi'], 'stoi_difference': q['stoi']-r['stoi']})
        if (i+1)%50 == 0:
            print('WAV audit', i+1, '/', len(pairs), flush=True)
    table(output/'wav_quality_per_file.csv', measurements)
    qualities = []
    for condition in ('SPEC_1', 'SPEC_10'):
        chosen = [r for r in measurements if r['condition']==condition]
        qualities.append({'condition': condition, 'n': len(chosen),
            'mean_stoi': float(np.mean([r['stoi'] for r in chosen])),
            'min_stoi': min(r['stoi'] for r in chosen),
            'mean_snr_db': float(np.mean([r['snr_db'] for r in chosen])),
            'min_snr_db': min(r['snr_db'] for r in chosen),
            'max_linf': max(r['linf'] for r in chosen),
            'epsilon_failures': sum(not r['epsilon_pass'] for r in chosen),
            'strict_epsilon_exceedances': sum(not r['strict_epsilon_pass'] for r in chosen),
            'protected_rail_samples': sum(r['protected_rail_samples'] for r in chosen),
            'clean_rail_samples': sum(r['clean_rail_samples'] for r in chosen),
            'max_previous_stoi_difference': max(abs(r['stoi_difference']) for r in chosen)})
    table(output/'wav_quality_summary.csv', qualities)
    provenance = evaluation['evaluation_models']
    for name, expected in provenance['sha256'].items():
        verify(Path('/home/safespeech/models/melo-evaluation-20261118')/name, expected)
    table(output/'evaluation_prompts.csv', [{'sentence_id':r['utterance_id'], 'text':r['text'], 'features':r['features'], 'features_sha256':r['features_sha256'], 'seed':2026+i} for i,r in enumerate(test.values())])
    refs = [r for r in data if r['split']=='train'][:5]
    write(output/'audit.json', {'status':'PASS' if not errors else 'FAIL', 'errors':errors,
        'training':training, 'split_counts':dict(split), 'cer_summary':summary, 'quality_summary':qualities,
        'sim_references':[{'id':r['utterance_id'], 'wav':r['wav'], 'sha256':r['wav_sha256']} for r in refs],
        'evaluation':str(evaluation_path), 'evaluation_seed':2026,
        'cer_calculation':'macro mean of utterance (S+D+I)/N; corpus CER separately labelled',
        'cer_normalization':"re.sub(r'[^0-9A-Za-z가-힣]', '', text); no Unicode or case normalization",
        'sim_calculation':'mean ECAPA cosine over five fixed clean training references; then mean over prompts',
        'original_results_overwritten':False, 'hashes':evidence})
    print(json.dumps({'status':'PASS' if not errors else 'FAIL', 'errors':errors, 'cer':summary, 'quality':qualities}), flush=True)
    return errors


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    raise SystemExit(bool(audit(args.study, args.output)))
