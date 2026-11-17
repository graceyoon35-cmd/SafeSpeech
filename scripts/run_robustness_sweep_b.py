"""Repeat a paired transform manifest and index artifacts; no efficacy scoring."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import mkdtemp

from robustness_b import validate_condition
from run_robustness_b import ROOT, run_pair, sha256


def default_conditions():
    return ([{'name': 'gaussian_noise', 'seed': seed, 'std': std}
             for std in (.001, .005, .01) for seed in (42, 43, 44)]
            + [{'name': 'mp3', 'bitrate_kbps': bitrate} for bitrate in (32, 64, 128)]
            + [{'name': 'resample', 'target_hz': hz} for hz in (8000, 12000, 16000)])


def run_sweep(manifest_path, output_root=ROOT / 'runs', repeats=2):
    if type(repeats) is not int or not 2 <= repeats <= 10:
        raise ValueError('repeats must be an integer between 2 and 10')
    manifest_path = Path(manifest_path).resolve()
    source_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    # Normalize paths before taking a snapshot in a different directory.
    if not isinstance(manifest, dict) or not isinstance(manifest.get('audio'), dict):
        raise ValueError('manifest audio must be an object')
    for role, value in manifest['audio'].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError('audio paths must be nonempty strings')
        manifest['audio'][role] = str((manifest_path.parent / value).resolve(strict=True))
    conditions = manifest.get('conditions')
    if not isinstance(conditions, list) or not conditions:
        raise ValueError('conditions must be a nonempty list')
    for condition in conditions:
        validate_condition(condition)
    labels = [json.dumps(c, sort_keys=True, allow_nan=False) for c in conditions]
    if len(set(labels)) != len(labels):
        raise ValueError('duplicate conditions are ambiguous; use repeats instead')
    audio_hashes = {role: sha256(path) for role, path in manifest['audio'].items()}
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix='robustness_sweep_b_', dir=output_root))
    snapshot = directory / 'manifest.json'
    snapshot.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    snapshot_hash = sha256(snapshot)
    summary = {
        'schema_version': 1, 'status': 'RUNNING', 'scope': 'single_pair_transform_sweep_only',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_manifest': str(manifest_path), 'source_manifest_sha256': source_hash,
        'manifest': manifest, 'manifest_snapshot': 'manifest.json',
        'condition_count': len(conditions), 'repeats': repeats,
        'repeat_meaning': 'same-condition reproducibility check, not independent speech trials',
        'reports': [], 'wav_count': 0, 'mp3_count': 0,
        'reproducibility': [],
        'not_evaluated': ['STOI', 'SIM', 'MCD', 'WER', 'protection efficacy', 'multiple utterances'],
        'evaluation_note': 'No implicit alignment. Resampled files retain target Hz; agree reference policy before scoring.',
    }
    rows = []
    baseline = {}
    try:
        for repeat in range(1, repeats + 1):
            if sha256(snapshot) != snapshot_hash:
                raise RuntimeError('manifest snapshot changed during experiment')
            report_path, report = run_pair(snapshot, directory / f'repeat_{repeat:02d}')
            summary['reports'].append(str(report_path.relative_to(directory)))
            expected = {(i, role) for i in range(len(conditions)) for role in ('original', 'protected')}
            observed = set()
            for item in report['outputs']:
                key = (item['condition_index'], item['role'])
                if key in observed:
                    raise RuntimeError('duplicate output entry')
                observed.add(key)
                artifact = Path(item['path'])
                if sha256(artifact) != item['sha256']:
                    raise RuntimeError('output WAV hash differs from its report')
                compressed = item.get('mp3')
                if compressed and sha256(compressed['path']) != compressed['sha256']:
                    raise RuntimeError('output MP3 hash differs from its report')
                signature = (item['sha256'], compressed['sha256'] if compressed else None,
                             item['output_rate'], item['output_samples'])
                if repeat == 1:
                    baseline[key] = signature
                else:
                    summary['reproducibility'].append({
                        'repeat': repeat, 'condition_index': key[0], 'role': key[1],
                        'matches_first': signature == baseline.get(key),
                    })
                row = {
                    'repeat': repeat, 'condition_index': key[0], 'role': key[1],
                    'transform': item['parameters']['name'],
                    'parameters_json': json.dumps(item['parameters'], sort_keys=True),
                    'input_sha256': report['inputs'][key[1]]['sha256_before'],
                    'input_rate': item['input_rate'], 'output_rate': item['output_rate'],
                    'input_samples': item['input_samples'], 'output_samples': item['output_samples'],
                    'input_seconds': item['input_seconds'], 'output_seconds': item['output_seconds'],
                    'duration_delta_seconds': item['output_seconds'] - item['input_seconds'],
                    'clipped_samples': item['clipped_samples'],
                    'wav_path': str(artifact.relative_to(directory)), 'wav_sha256': item['sha256'],
                    'mp3_path': str(Path(compressed['path']).relative_to(directory)) if compressed else '',
                    'mp3_sha256': compressed['sha256'] if compressed else '',
                    'ffmpeg_version': item.get('ffmpeg_version', ''),
                    'report_path': str(report_path.relative_to(directory)),
                }
                rows.append(row)
            if observed != expected:
                raise RuntimeError('missing or unexpected output entries')
            summary['wav_count'] = len(rows)
            summary['mp3_count'] = sum(bool(row['mp3_path']) for row in rows)
        if not all(item['matches_first'] for item in summary['reproducibility']):
            raise RuntimeError('same-condition artifact reproducibility check failed')
        summary['reproducibility_pass'] = True
        summary['input_files_unchanged'] = all(
            sha256(path) == audio_hashes[role] for role, path in manifest['audio'].items())
        if not summary['input_files_unchanged']:
            raise RuntimeError('input file changed between repetitions')
        if sha256(manifest_path) != source_hash or sha256(snapshot) != snapshot_hash:
            raise RuntimeError('manifest changed during experiment')
        summary['status'] = 'PASS'
    except Exception as error:
        summary['status'] = 'FAIL'
        summary['error'] = str(error)
        raise
    finally:
        summary['wav_count'] = len(rows)
        summary['mp3_count'] = sum(bool(row['mp3_path']) for row in rows)
        if rows:
            with (directory / 'artifacts.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (directory / 'summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return directory / 'summary.json', summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    path, summary = run_sweep(args.manifest, args.output_root, args.repeats)
    print(json.dumps({key: summary[key] for key in (
        'status', 'condition_count', 'repeats', 'wav_count', 'mp3_count',
        'reproducibility_pass', 'input_files_unchanged')}
        | {'summary': str(path), 'csv': str(path.parent / 'artifacts.csv')}, indent=2))


if __name__ == '__main__':
    main()
