"""Create machine-readable, explicitly scoped tables from actual desktop artifacts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from run_korean_melo_spec import save, sha256, validate_handoff


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--spec-run', type=Path, required=True)
    parser.add_argument('--robustness-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = args.audit_dir / 'data/data_set/kss_part_200'
    manifest = read(data / 'manifest.json')
    spec_input = read(data / 'spec_input.json')['utterances']
    checks = list(csv.DictReader((data / 'transcript_check.csv').open(encoding='utf-8-sig')))
    rows = manifest['utterances']
    by_id = {row['utterance_id']: row for row in rows}
    assert len(by_id) == len(rows) == len(spec_input) == len(checks) == 200
    for row in spec_input:
        original = by_id[row['utterance_id']]
        assert row['text'] == original['transcript']
        assert row['split'] == original['split']
        assert row['original_sha256'] == original['sha256'] == sha256(data / row['original_path'])
    for split in ('train', 'validation', 'test'):
        expected = {r['utterance_id'] for r in rows if r['split'] == split}
        lines = (data / 'splits' / f'{split}.txt').read_text().split()
        assert len(lines) == len(set(lines)) and set(lines) == expected
    data_summary = {'wav_count': len(list(data.rglob('*.wav'))),
                    'splits': dict(Counter(r['split'] for r in rows)),
                    'manifest_spec_split_hash_consistency': 'PASS',
                    'usage_status': manifest['dataset']['usage_status'],
                    'research_use_permitted_count': sum(r['research_use_permitted'] is True for r in spec_input),
                    'transcript_verified_count': sum(r['transcript_verified'] is True for r in spec_input),
                    'human_experiment_ready': False}
    handoff = read(args.spec_run / 'handoff_manifest.json')
    validate_handoff(handoff['utterances'])
    measurement_rows = []
    for row in handoff['utterances']:
        case = Path(row['validation_path']).parent
        report = read(case / 'report.json')['files'][0]
        gradient = read(case / 'gradients.json')
        assert len(gradient) == row['iterations']
        assert all(g['finite'] and g['nonzero'] > 0 for g in gradient)
        measurement_rows.append({'input_kind': row['input_kind'], 'iterations': row['iterations'],
            'audio_seconds': report['audio_seconds'], 'generation_seconds': report['seconds'],
            'max_abs_delta_before_pcm': report['max_abs_delta'],
            'epsilon': row['epsilon'], 'gradient_steps_verified': len(gradient),
            'stoi_numerical_only_not_speech_quality': row['stoi'],
            'claim_scope': 'Mathematical fixture; no Korean speech or cloning-defense efficacy'})
    batch = read(args.robustness_summary)
    assert batch['status'] == 'PASS'
    artifact_count = 0
    for pair in batch['pairs']:
        summary_path = args.robustness_summary.parent / pair['summary']
        summary = read(summary_path)
        assert summary['reproducibility_pass'] and summary['input_files_unchanged']
        for artifact in csv.DictReader((summary_path.parent / 'artifacts.csv').open(encoding='utf-8-sig')):
            assert sha256(summary_path.parent / artifact['wav_path']) == artifact['wav_sha256']
            if artifact['mp3_path']:
                assert sha256(summary_path.parent / artifact['mp3_path']) == artifact['mp3_sha256']
            artifact_count += 1
    summary = {'kss': data_summary, 'spec': read(args.spec_run / 'run.json'),
               'gpu_memory': read(args.spec_run / 'gpu_memory.json'),
               'robustness': {'status': batch['status'], 'conditions': len(batch['conditions']),
                   'input_pairs': batch['input_pair_count'], 'wav_count': batch['wav_count'],
                   'mp3_count': batch['mp3_count'], 'hash_verified_wav_count': artifact_count,
                   'repeats': batch['repeats'], 'scope': 'mock input transforms; not defense efficacy'},
               'not_performed': ['Human SPEC', 'Korean listening verification', 'Finetuning', 'SIM', 'CER']}
    save(args.output / 'summary.json', summary)
    with (args.output / 'technical_measurements.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurement_rows[0]))
        writer.writeheader()
        writer.writerows(measurement_rows)
    print(json.dumps({'kss': data_summary, 'gpu_memory': summary['gpu_memory'],
                      'robustness': summary['robustness']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
