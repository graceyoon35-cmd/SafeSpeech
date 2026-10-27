"""Verify a completed sweep and export a portable, score-free evaluation handoff."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import shutil
from tempfile import mkdtemp
import zipfile

import soundfile as sf

from run_robustness_b import ROOT, sha256


def child(root, relative):
    # Package paths are always relative and may never escape their root.
    if not isinstance(relative, str):
        raise ValueError('expected a relative POSIX path')
    # Windows experiment CSVs use backslashes; export still writes POSIX paths.
    relative = relative.replace('\\', '/')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or not path.parts or ':' in relative:
        raise ValueError('unsafe relative path')
    resolved = (root / str(path)).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError('path escapes package root')
    return resolved


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_handoff(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'package_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'PASS' or not manifest.get('files'):
        raise ValueError('package manifest is not complete')
    for relative, digest in manifest['files'].items():
        path = child(directory, relative)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'package file failed checksum: {relative}')
    return manifest


def export_handoff(summary_path, output_root=ROOT / 'runs'):
    summary_path = Path(summary_path).resolve()
    source = summary_path.parent
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('status') != 'PASS' or summary.get('reproducibility_pass') is not True or summary.get('input_files_unchanged') is not True:
        raise ValueError('only a successful, reproducible sweep can be exported')
    with (source / 'artifacts.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    conditions = summary['manifest']['conditions']
    repeats = summary['repeats']
    expected = {(repeat, i, role) for repeat in range(1, repeats + 1)
                for i in range(len(conditions)) for role in ('original', 'protected')}
    if len(rows) != len(expected) or len(rows) != summary['wav_count']:
        raise ValueError('artifact count differs from experiment settings')
    inputs = {}
    first_report = json.loads(child(source, summary['reports'][0]).read_text(encoding='utf-8'))
    for role in ('original', 'protected'):
        path = Path(summary['manifest']['audio'][role])
        expected_hash = first_report['inputs'][role]['sha256_before']
        if sha256(path) != expected_hash:
            raise ValueError(f'input changed since experiment: {role}')
        inputs[role] = (path, expected_hash)
    checked = []
    observed = set()
    seen_destinations = set()
    for row in rows:
        key = (int(row['repeat']), int(row['condition_index']), row['role'])
        if key not in expected or key in observed:
            raise ValueError('duplicate or unexpected condition/role/repeat')
        observed.add(key)
        if json.loads(row['parameters_json']) != conditions[key[1]]:
            raise ValueError('CSV parameters do not match the experiment manifest')
        if row['input_sha256'] != inputs[key[2]][1]:
            raise ValueError('CSV input checksum does not match the experiment input')
        is_mp3 = conditions[key[1]]['name'] == 'mp3'
        if bool(row['mp3_path']) != is_mp3:
            raise ValueError('compressed artifact is missing or unexpected')
        wav = child(source, row['wav_path'])
        if str(wav) in seen_destinations:
            raise ValueError('multiple entries use the same WAV')
        seen_destinations.add(str(wav))
        if sha256(wav) != row['wav_sha256']:
            raise ValueError('WAV differs from experiment checksum')
        info = sf.info(wav)
        if info.channels != 1 or info.samplerate != int(row['output_rate']) or info.frames != int(row['output_samples']):
            raise ValueError('WAV metadata differs from CSV')
        mp3 = child(source, row['mp3_path']) if row['mp3_path'] else None
        if mp3 and sha256(mp3) != row['mp3_sha256']:
            raise ValueError('MP3 differs from experiment checksum')
        checked.append((row, wav, mp3))
    if observed != expected:
        raise ValueError('missing artifact rows')
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix='handoff_b_', dir=output_root))
    # Only verified files are copied; the source experiment is never edited.
    exported_rows = []
    for role, (path, digest) in inputs.items():
        target = directory / 'inputs' / f'{role}.wav'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(target) != digest:
            raise RuntimeError('input copy failed checksum')
    lookup = {}
    for row, wav, mp3 in checked:
        repeat, index, role = int(row['repeat']), int(row['condition_index']), row['role']
        folder = directory / 'results' / f'condition_{index:02d}' / f'repeat_{repeat:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f'{role}.wav'
        shutil.copyfile(wav, target)
        if sha256(target) != row['wav_sha256']:
            raise RuntimeError('artifact copy failed checksum')
        entry = {key: value for key, value in row.items() if key not in ('wav_path', 'mp3_path', 'report_path')}
        entry['wav_path'] = target.relative_to(directory).as_posix()
        entry['mp3_path'] = ''
        if mp3:
            compressed = folder / f'{role}.mp3'
            shutil.copyfile(mp3, compressed)
            if sha256(compressed) != row['mp3_sha256']:
                raise RuntimeError('MP3 copy failed checksum')
            entry['mp3_path'] = compressed.relative_to(directory).as_posix()
        exported_rows.append(entry)
        lookup[(repeat, index, role)] = entry
    pairs = []
    for repeat in range(1, repeats + 1):
        for index, condition in enumerate(conditions):
            original = lookup[(repeat, index, 'original')]
            protected = lookup[(repeat, index, 'protected')]
            if original['output_rate'] != protected['output_rate'] or original['output_samples'] != protected['output_samples']:
                raise ValueError('transformed pair is not strictly aligned')
            pairs.append({
                'pair_id': f'condition_{index:02d}_repeat_{repeat:02d}',
                'condition_index': index, 'repeat': repeat,
                'parameters_json': json.dumps(condition, sort_keys=True),
                'transformed_original': original['wav_path'],
                'transformed_protected': protected['wav_path'],
                'input_original': 'inputs/original.wav', 'input_protected': 'inputs/protected.wav',
                'sample_rate': original['output_rate'], 'samples': original['output_samples'],
                'evaluation_status': 'not_measured', 'reference_policy': 'confirm_with_evaluator',
            })
    write_csv(directory / 'artifacts.csv', exported_rows)
    write_csv(directory / 'evaluation_pairs.csv', pairs)
    portable = {'schema_version': 1, 'experiment_id': summary['manifest']['experiment_id'],
                'audio': {role: f'inputs/{role}.wav' for role in inputs}, 'conditions': conditions}
    write_json(directory / 'experiment.json', portable)
    write_json(directory / 'verification.json', {
        'status': 'PASS', 'scope': 'transform_and_reproducibility_only',
        'source_summary_sha256': sha256(summary_path),
        'condition_count': len(conditions), 'repeats': repeats,
        'wav_count': len(exported_rows), 'mp3_count': sum(bool(row['mp3_path']) for row in exported_rows),
        'pair_count': len(pairs), 'independent_utterance_count': 1,
        'reproducibility_pass': True, 'input_files_unchanged': True,
        'versions': first_report['versions'], 'python': first_report['python'],
        'platform': first_report['platform'],
        'not_evaluated': summary['not_evaluated'],
    })
    (directory / 'README_KO.md').write_text('''# 2학년 B 평가 전달 자료

이 폴더 전체를 내려받아 압축 해제하세요. 모든 파일 경로는 이 폴더 기준입니다.

- inputs/: 실험에 사용한 원본 및 실제 SPEC 보호 음성
- results/: 조건별·회차별 변환 WAV와 MP3
- experiment.json: 변환 조건과 원본/보호 파일 경로
- artifacts.csv: 파일별 조건·전후 Hz/길이·해시·클리핑 수
- evaluation_pairs.csv: 동일 변환 조건의 원본/보호 파일 쌍과 입력 음성 경로
- verification.json: 변환 및 재현성 확인 결과
- package_manifest.json: 전달 파일의 SHA-256

실제 발화는 한 쌍입니다. 같은 조건의 반복은 재현성 확인이며 독립 발화 표본이 아닙니다.
아직 STOI/SIM/MCD/WER를 측정하지 않았습니다. 미측정값을 0으로 기록하지 마세요.

evaluation_pairs.csv는 평가할 파일을 찾는 목록입니다. 최종 reference 정책은
평가 담당자와 합의해야 합니다. 동일조건 변환 원본/보호는 서로 Hz·길이가 일치합니다.
변환 전 입력을 기준으로 비교하려면 특히 리샘플링 조건의 Hz·길이를 명시적으로
맞춰야 합니다. 자동 잘라내기로 정렬했다고 간주하지 마세요.
STOI가 높거나 변환이 성공했다는 사실만으로 보호 효과가 검증되지는 않습니다.

공개 체크포인트의 화자 임베딩 누락 등 기존 SPEC 생성 제약은 별도로 유지됩니다.
보고된 보호 효과 결과가 아니라 실행 가능한 비교 자료입니다.
''', encoding='utf-8')
    files = {path.relative_to(directory).as_posix(): sha256(path)
             for path in sorted(directory.rglob('*')) if path.is_file()}
    write_json(directory / 'package_manifest.json', {'schema_version': 1, 'status': 'PASS', 'files': files})
    verify_handoff(directory)
    archive = directory.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                zipped.write(path, f'{directory.name}/{path.relative_to(directory).as_posix()}')
    return directory, archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    directory, archive = export_handoff(args.summary, args.output_root)
    print(json.dumps({'status': 'PASS', 'folder': str(directory), 'zip': str(archive),
                      'zip_sha256': sha256(archive)}, indent=2))


if __name__ == '__main__':
    main()
