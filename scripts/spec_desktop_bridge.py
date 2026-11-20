"""Standard-library desktop client for the pinned SPEC CLI (no Qt/torch import)."""
from array import array
from dataclasses import dataclass
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time
import uuid
import wave


class SpecIntegrationError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def windows_to_wsl(value):
    """Map a local drive path without invoking a shell; UNC is unsupported."""
    path = PureWindowsPath(value)
    if not path.is_absolute() or len(path.drive) != 2 or path.drive[1] != ':':
        raise ValueError('WSL 연결에는 C:\\... 형태의 로컬 드라이브 절대 경로가 필요합니다.')
    if any(c in str(path) for c in ('\0', '\r', '\n')):
        raise ValueError('Invalid path characters')
    return '/mnt/' + path.drive[0].lower() + '/' + '/'.join(path.parts[1:])


def pcm16(path):
    try:
        with wave.open(str(path), 'rb') as stream:
            channels, width, rate, samples = (stream.getnchannels(), stream.getsampwidth(),
                                             stream.getframerate(), stream.getnframes())
            if (channels, width, rate, stream.getcomptype()) != (1, 2, 44100, 'NONE'):
                raise ValueError('mono 44,100Hz PCM16 WAV가 필요합니다. 자동 변환하지 않습니다.')
            if not 44100 <= samples <= 441000:
                raise ValueError('SPEC 입력은 1~10초 WAV여야 합니다. 자동 분할하지 않습니다.')
            raw = stream.readframes(samples)
    except (wave.Error, EOFError) as error:
        raise ValueError('정상적인 PCM16 WAV 파일이 필요합니다.') from error
    if len(raw) != 2 * samples:
        raise ValueError('WAV 샘플 데이터가 잘렸습니다.')
    values = array('h', raw)
    if sys.byteorder != 'little':
        values.byteswap()
    if not any(values):
        raise ValueError('완전 무음 WAV는 보호할 수 없습니다.')
    return values


@dataclass(frozen=True)
class SpecConfig:
    repository: Path
    output_root: Path
    execution: str = 'auto'  # Windows -> WSL; Linux/WSL -> local subprocess.
    distribution: str = 'Ubuntu-22.04'
    python_executable: str = '/home/safespeech/.venvs/melo-training-20261118/bin/python'
    assets: str = '/home/safespeech/models/melo-spec-pinned-20261118'
    iterations: int = 10
    epsilon: float = 8.0
    seed: int = 52


@dataclass(frozen=True)
class SpecDesktopResult:
    original_path: Path
    elapsed_seconds: float
    output_path: Path
    metadata_path: Path
    diagnostics_directory: Path
    confidence: None = None  # No calibrated defense probability exists.

    @property
    def protected_name(self):
        return self.original_path.stem + '_protected.wav'

    def as_dict(self):
        return {'status': 'PASS', 'original_path': str(self.original_path),
                'elapsed_seconds': self.elapsed_seconds, 'output_path': str(self.output_path),
                'protected_name': self.protected_name, 'metadata_path': str(self.metadata_path),
                'diagnostics_directory': str(self.diagnostics_directory), 'confidence': None}


class SpecClient:
    """Blocking call: use the Qt processor or another worker off the GUI thread."""
    def __init__(self, config):
        self.config = config
        if config.execution not in ('auto', 'wsl', 'local'):
            raise ValueError('execution must be auto, wsl, or local')
        if type(config.iterations) is not int or not 1 <= config.iterations <= 200:
            raise ValueError('iterations must be 1..200')
        if not math.isfinite(config.epsilon) or not 0 < config.epsilon <= 255:
            raise ValueError('epsilon must be in (0,255]')
        if type(config.seed) is not int or not 0 <= config.seed < 2 ** 32:
            raise ValueError('seed must be uint32')

    @property
    def uses_wsl(self):
        return self.config.execution == 'wsl' or (self.config.execution == 'auto' and os.name == 'nt')

    def command(self, source, destination):
        cfg = self.config
        script = Path(cfg.repository).resolve() / 'scripts/protect_wav.py'
        if self.uses_wsl:
            command = ['wsl.exe', '--distribution', cfg.distribution, '--exec', cfg.python_executable]
            script, source, destination = map(windows_to_wsl, (script, source, destination))
        else:
            command = [cfg.python_executable]
        return command + [str(script), '--input', str(source), '--output-dir', str(destination),
                          '--assets', cfg.assets, '--device', 'cuda', '--iterations', str(cfg.iterations),
                          '--epsilon', str(cfg.epsilon), '--seed', str(cfg.seed)]

    def protect(self, source):
        started = time.perf_counter()
        source = Path(getattr(source, 'path', source)).expanduser().resolve(strict=True)
        if not source.is_file() or source.suffix.lower() != '.wav':
            raise ValueError('현재 SPEC 연결은 WAV만 지원합니다. MP3/MP4는 앱에서 별도 처리해야 합니다.')
        original = pcm16(source)
        before = sha256(source)
        root = Path(self.config.output_root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        job = root / ('spec-' + uuid.uuid4().hex)
        job.mkdir(exist_ok=False)
        destination = job / 'result'  # Must not exist: protect_wav refuses overwrite.
        command = self.command(source, destination)
        (job / 'request.json').write_text(json.dumps({'schema_version': 1, 'command': command,
             'input': str(source), 'input_sha256': before}, ensure_ascii=False, indent=2), encoding='utf-8')
        try:
            proc = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                                  errors='replace', shell=False,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name=='nt' else 0)
            (job / 'stdout.log').write_text(proc.stdout, encoding='utf-8')
            (job / 'stderr.log').write_text(proc.stderr, encoding='utf-8')
            if proc.returncode != 0:
                raise SpecIntegrationError(f'SPEC 종료 코드 {proc.returncode}. 진단: {job}')
            try:
                reply = json.loads(proc.stdout.strip().splitlines()[-1])
                saved = json.loads((destination / 'result.json').read_text(encoding='utf-8'))
            except (IndexError, ValueError, OSError) as error:
                raise SpecIntegrationError('SPEC 성공 응답 또는 result.json을 읽을 수 없습니다.') from error
            if not isinstance(reply, dict) or not isinstance(saved, dict) or reply != saved or saved.get('status') != 'PASS':
                raise SpecIntegrationError('SPEC 응답과 저장 결과가 일치하는 PASS가 아닙니다.')
            output = destination / 'protected.wav'
            expected_output = windows_to_wsl(output) if self.uses_wsl else str(output)
            expected_input = windows_to_wsl(source) if self.uses_wsl else str(source)
            if saved.get('output') != expected_output or saved.get('input') != expected_input:
                raise SpecIntegrationError('SPEC 결과가 요청한 입출력 경로와 다릅니다.')
            if output.is_symlink() or sha256(output) != saved.get('output_sha256'):
                raise SpecIntegrationError('보호 WAV 해시가 일치하지 않습니다.')
            if sha256(source) != before or saved.get('input_sha256') != before:
                raise SpecIntegrationError('원본 WAV 해시가 변경됐습니다.')
            if (saved.get('iterations'), saved.get('epsilon_units_255'), saved.get('seed')) != (
                    self.config.iterations, self.config.epsilon, self.config.seed):
                raise SpecIntegrationError('SPEC 실행 설정이 요청과 다릅니다.')
            protected = pcm16(output)
            if len(protected) != len(original) or saved.get('samples') != len(original):
                raise SpecIntegrationError('보호 WAV 길이가 원본과 다릅니다.')
            measured = max(abs(a-b) for a,b in zip(original, protected)) / 32768
            if not measured <= self.config.epsilon/255:
                raise SpecIntegrationError('보호 WAV가 엄격한 ε 제한을 초과합니다.')
            if saved.get('strict_linf') != measured or saved.get('sample_rate') != 44100 or saved.get('subtype') != 'PCM_16':
                raise SpecIntegrationError('보호 WAV의 실제 형식·ε와 진단 값이 다릅니다.')
            result = SpecDesktopResult(source, time.perf_counter()-started, output,
                                       destination/'result.json', job)
            (job/'desktop_result.json').write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
            return result
        except Exception as error:
            (job/'desktop_failure.json').write_text(json.dumps({'status':'FAIL','error_type':type(error).__name__,
                'error':str(error)},ensure_ascii=False,indent=2),encoding='utf-8')
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--execution', choices=['auto','wsl','local'], default='auto')
    p.add_argument('--distribution', default='Ubuntu-22.04')
    p.add_argument('--python-executable', default=SpecConfig.python_executable)
    p.add_argument('--assets', default=SpecConfig.assets)
    args = p.parse_args()
    try:
        client = SpecClient(SpecConfig(args.repository, args.output_root, args.execution,
            args.distribution, args.python_executable, args.assets))
        result = client.protect(args.input)
    except Exception as error:
        print(json.dumps({'status':'FAIL','error_type':type(error).__name__,'error':str(error)},ensure_ascii=False))
        return 1
    print(json.dumps(result.as_dict(),ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
