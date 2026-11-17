"""Run existing paired sweep manifests together and build a combined artifact index."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import mkdtemp

from robustness_b import validate_condition
from run_robustness_b import ROOT, sha256
from run_robustness_sweep_b import run_sweep


def run_batch(manifest_paths, output_root=ROOT / 'runs', repeats=2):
    if type(repeats) is not int or not 2 <= repeats <= 10:
        raise ValueError('repeats must be an integer between 2 and 10')
    paths = [Path(path).resolve() for path in manifest_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('provide at least one manifest and no duplicate manifest paths')

    manifests = []
    ids = set()
    input_pairs = set()
    conditions = None
    for path in paths:
        payload = path.read_bytes()
        manifest = json.loads(payload)
        if (not isinstance(manifest, dict)
                or set(manifest) != {'schema_version', 'experiment_id', 'audio', 'conditions'}
                or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1):
            raise ValueError(f'{path.name}: expected a version 1 pair manifest')
        experiment_id = manifest['experiment_id']
        if not isinstance(experiment_id, str) or not experiment_id.strip() or experiment_id in ids:
            raise ValueError(f'{path.name}: experiment_id must be nonempty and unique')
        ids.add(experiment_id)
        audio = manifest['audio']
        if (not isinstance(audio, dict) or set(audio) != {'original', 'protected'}
                or any(not isinstance(value, str) or not value.strip() for value in audio.values())):
            raise ValueError(f'{path.name}: provide original and protected paths')
        manifest['audio'] = {role: str((path.parent / value).resolve(strict=True))
                             for role, value in audio.items()}
        hashes = tuple(sha256(manifest['audio'][role]) for role in ('original', 'protected'))
        if hashes[0] == hashes[1] or hashes in input_pairs:
            raise ValueError(f'{path.name}: identical inputs or repeated input pair')
        input_pairs.add(hashes)
        current = manifest['conditions']
        if not isinstance(current, list) or not current:
            raise ValueError(f'{path.name}: conditions must be a nonempty list')
        for condition in current:
            validate_condition(condition)
        if len({json.dumps(c, sort_keys=True) for c in current}) != len(current):
            raise ValueError(f'{path.name}: conditions must be unique')
        if conditions is not None and conditions != current:
            raise ValueError(f'{path.name}: use the same condition list and order throughout the batch')
        conditions = current
        manifests.append((manifest, hashes))

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix='robustness_batch_b_', dir=output_root))
    report = {'status': 'RUNNING', 'scope': 'paired_transforms_only', 'repeats': repeats,
              'conditions': conditions, 'input_pair_count': len(paths),
              'distinct_original_sha256_count': len({pair[0] for pair in input_pairs}),
              'pairs': [], 'wav_count': 0, 'mp3_count': 0}
    rows = []
    try:
        for index, (manifest, hashes) in enumerate(manifests, 1):
            experiment_id = manifest['experiment_id']
            pair_root = directory / f'pair_{index:04d}'
            pair_root.mkdir()
            snapshot = pair_root / 'input.json'
            snapshot.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                encoding='utf-8')
            if any(sha256(manifest['audio'][role]) != digest
                   for role, digest in zip(('original', 'protected'), hashes)):
                raise ValueError(f'{experiment_id}: input changed after batch validation')
            summary_path, summary = run_sweep(snapshot, pair_root, repeats)
            with (summary_path.parent / 'artifacts.csv').open(encoding='utf-8-sig', newline='') as stream:
                pair_rows = list(csv.DictReader(stream))
            expected_hashes = dict(zip(('original', 'protected'), hashes))
            if any(row['input_sha256'] != expected_hashes[row['role']] for row in pair_rows):
                raise ValueError(f'{experiment_id}: input changed before the sweep')
            for row in pair_rows:
                for key in ('wav_path', 'mp3_path', 'report_path'):
                    if row[key]:
                        row[key] = (summary_path.parent / row[key]).relative_to(directory).as_posix()
                rows.append({'experiment_id': experiment_id, **row})
            report['pairs'].append({'experiment_id': experiment_id,
                                    'summary': summary_path.relative_to(directory).as_posix()})
            report['wav_count'] += summary['wav_count']
            report['mp3_count'] += summary['mp3_count']
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL', error=str(error))
        raise
    finally:
        if rows:
            with (directory / 'artifacts.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (directory / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
    return directory / 'summary.json', report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifests', nargs='+', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    path, report = run_batch(args.manifests, args.output_root, args.repeats)
    print(json.dumps({'summary': str(path), **report}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
