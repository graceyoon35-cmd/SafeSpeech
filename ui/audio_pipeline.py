"""
audio_pipeline.py  -  영상/음성 파일 처리 파이프라인 (PC 앱용)

흐름
    입력 검증(ffprobe) -> 오디오 추출(ffmpeg) -> 청크 분할 -> [모델: ChunkProcessor]
    -> 청크 병합(crossfade) -> 영상 재결합(ffmpeg, 영상은 복사) -> 저장 -> 임시파일 정리

의존성: Python 3.9+, numpy, ffmpeg/ffprobe 실행파일 (앱에 번들 권장)
설계 메모
  * 오디오는 디스크의 raw 파일(s16le)로 두고, 청크 단위로만 읽어서 메모리를 아낀다.
  * 청크 결과는 chunks_out/*.npy 로 즉시 저장 -> 중단 후 재실행하면 완료된 청크는 건너뛴다.
  * 모델은 ChunkProcessor 인터페이스 하나로만 연결된다 (모델팀 코드는 여기만 구현).
"""
from __future__ import annotations

import enum
import errno
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence, Tuple

import numpy as np

log = logging.getLogger("audio_pipeline")


# ----------------------------------------------------------------------------
# 1. 설정
# ----------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    # ffmpeg 위치 (앱에 번들했다면 번들 경로를 지정)
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    # 모델 입력 규격 (모델팀과 확정 필요)
    model_sample_rate: int = 16000

    # 청크 설정: 3~8초 기본, 최대 10초
    chunk_sec: float = 8.0          # 일반 청크 길이
    min_chunk_sec: float = 3.0      # 마지막 청크 최소 길이 (이보다 짧으면 앞쪽으로 당겨서 3초 확보)
    max_chunk_sec: float = 10.0     # 상한 (chunk_sec 이 이를 넘을 수 없음)
    overlap_sec: float = 0.5        # 청크 간 겹침
    crossfade_sec: float = 0.5      # 겹침 구간 중 crossfade 길이 (<= overlap_sec)

    # 배치 / 재시도
    batch_size: int = 4             # OOM 발생 시 자동으로 줄어듦
    max_retries: int = 3
    retry_backoff_sec: float = 1.0

    # 입력 제한
    max_duration_sec: float = 3 * 3600 + 600   # 3시간 10분
    max_file_bytes: Optional[int] = None       # 예: 30 * 1024**3
    video_exts: Tuple[str, ...] = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".flv", ".wmv", ".ts")
    audio_exts: Tuple[str, ...] = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma")

    # 경로
    work_root: Path = field(default_factory=lambda: Path(tempfile.gettempdir()) / "voiceshield_jobs")
    output_dir: Optional[Path] = None          # None 이면 원본 파일과 같은 폴더
    output_suffix: str = "_protected"

    # 출력 오디오
    audio_bitrate: str = "192k"                # AAC 인코딩 비트레이트 (섭동 보존을 위해 높게)
    restore_source_format: bool = True         # 원본 샘플레이트/채널(최대 2ch)로 복원

    # 동작
    cleanup_on_success: bool = True
    ffmpeg_stall_timeout_sec: float = 180.0    # ffmpeg 출력이 이 시간 동안 없으면 hang 으로 보고 종료
    ffmpeg_total_timeout_sec: float = 6 * 3600
    disk_margin_bytes: int = 500 * 1024 ** 2

    def __post_init__(self):
        self.work_root = Path(self.work_root)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if not (0 < self.min_chunk_sec <= self.chunk_sec <= self.max_chunk_sec <= 10.0):
            raise ValueError("0 < min_chunk_sec <= chunk_sec <= max_chunk_sec <= 10 이어야 합니다")
        if not (0 < self.overlap_sec < self.chunk_sec / 2):
            raise ValueError("0 < overlap_sec < chunk_sec/2 이어야 합니다")
        if not (0 < self.crossfade_sec <= self.overlap_sec):
            raise ValueError("0 < crossfade_sec <= overlap_sec 이어야 합니다")
        if self.batch_size < 1:
            raise ValueError("batch_size >= 1")


# ----------------------------------------------------------------------------
# 2. 에러 정의 (UI 에서 코드 -> 메시지 매핑)
# ----------------------------------------------------------------------------
class ErrorCode(str, enum.Enum):
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    NO_AUDIO_STREAM = "NO_AUDIO_STREAM"
    TOO_LONG = "TOO_LONG"
    TOO_LARGE = "TOO_LARGE"
    LOW_STORAGE = "LOW_STORAGE"
    FFMPEG_NOT_FOUND = "FFMPEG_NOT_FOUND"
    FFMPEG_FAILED = "FFMPEG_FAILED"
    FFMPEG_TIMEOUT = "FFMPEG_TIMEOUT"
    MODEL_ERROR = "MODEL_ERROR"
    CHUNK_CORRUPT = "CHUNK_CORRUPT"
    LENGTH_MISMATCH = "LENGTH_MISMATCH"
    OUTPUT_PATH_ERROR = "OUTPUT_PATH_ERROR"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


USER_MESSAGES = {
    ErrorCode.FILE_NOT_FOUND: "파일을 찾을 수 없습니다.",
    ErrorCode.UNSUPPORTED_FORMAT: "지원하지 않거나 손상된 파일입니다.",
    ErrorCode.NO_AUDIO_STREAM: "오디오가 없는 파일입니다.",
    ErrorCode.TOO_LONG: "파일이 너무 깁니다. (최대 길이 초과)",
    ErrorCode.TOO_LARGE: "파일 용량이 너무 큽니다.",
    ErrorCode.LOW_STORAGE: "저장 공간이 부족합니다.",
    ErrorCode.FFMPEG_NOT_FOUND: "FFmpeg 를 찾을 수 없습니다. 앱을 다시 설치해 주세요.",
    ErrorCode.FFMPEG_FAILED: "미디어 변환 중 오류가 발생했습니다.",
    ErrorCode.FFMPEG_TIMEOUT: "미디어 변환이 응답하지 않아 중단했습니다.",
    ErrorCode.MODEL_ERROR: "보호 처리 중 오류가 발생했습니다.",
    ErrorCode.CHUNK_CORRUPT: "중간 결과 파일이 손상되었습니다. 다시 시도해 주세요.",
    ErrorCode.LENGTH_MISMATCH: "결과 길이가 원본과 일치하지 않아 중단했습니다.",
    ErrorCode.OUTPUT_PATH_ERROR: "저장 경로에 쓸 수 없습니다.",
    ErrorCode.CANCELLED: "작업이 취소되었습니다.",
    ErrorCode.UNKNOWN: "알 수 없는 오류가 발생했습니다.",
}


class PipelineError(Exception):
    def __init__(self, code: ErrorCode, message: str = "", detail: str = ""):
        super().__init__(f"[{code.value}] {message}")
        self.code = code
        self.message = message or USER_MESSAGES.get(code, "")
        self.detail = detail

    @property
    def user_message(self) -> str:
        return USER_MESSAGES.get(self.code, self.message)


# ----------------------------------------------------------------------------
# 3. 모델 인터페이스 (모델팀과 맞닿는 유일한 지점)
# ----------------------------------------------------------------------------
class ChunkProcessor(Protocol):
    """
    tag: 모델 이름/버전 문자열. 바뀌면 이전 작업의 중간 결과를 재사용하지 않는다.
    process_batch: float32 1차원 배열 리스트(-1.0~1.0, model_sample_rate, mono)를 받아
                   '같은 길이'의 배열 리스트를 반환해야 한다.
                   (마지막 청크는 길이가 다를 수 있음 -> 고정 길이 모델이면 FixedLengthAdapter 사용)
    OOM 이 나면 MemoryError 또는 메시지에 'out of memory' 가 포함된 예외를 던지면
    파이프라인이 배치를 자동으로 절반으로 줄여 재시도한다.
    """
    tag: str

    def process_batch(self, chunks: List[np.ndarray]) -> List[np.ndarray]: ...


class DummyProcessor:
    """모델 개발 전 테스트용: 입력을 그대로 반환."""
    tag = "dummy"

    def process_batch(self, chunks: List[np.ndarray]) -> List[np.ndarray]:
        return [c.copy() for c in chunks]


class FixedLengthAdapter:
    """
    입력 길이가 고정된 모델용 어댑터. 짧은 청크는 0 으로 패딩하고 결과를 원래 길이로 잘라낸다.
        fn: (ndarray[B, fixed_len]) -> ndarray[B, fixed_len]
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], fixed_len: int, tag: str = "model"):
        self.fn, self.fixed_len, self.tag = fn, fixed_len, tag

    def process_batch(self, chunks: List[np.ndarray]) -> List[np.ndarray]:
        batch = np.zeros((len(chunks), self.fixed_len), dtype=np.float32)
        for i, c in enumerate(chunks):
            if len(c) > self.fixed_len:
                raise ValueError("청크가 모델 고정 길이보다 깁니다")
            batch[i, : len(c)] = c
        out = np.asarray(self.fn(batch))
        return [out[i, : len(c)].astype(np.float32) for i, c in enumerate(chunks)]


# ----------------------------------------------------------------------------
# 4. 청크 계획 / 병합 (순수 함수 -> 단위 테스트 가능)
# ----------------------------------------------------------------------------
def plan_chunks(total: int, sr: int, chunk_sec: float, min_chunk_sec: float, overlap_sec: float
                ) -> List[Tuple[int, int]]:
    """
    (start, end) 샘플 구간 리스트.
    - 일반 청크: chunk_sec 길이, overlap_sec 만큼 겹침
    - 마지막 청크: 남은 길이가 min_chunk_sec 보다 짧으면 시작점을 앞으로 당겨 min_chunk_sec 를 확보
    - 전체가 min_chunk_sec 보다 짧으면 청크 1개 (모델 어댑터가 패딩)
    """
    if total <= 0:
        return []
    chunk_len = int(round(chunk_sec * sr))
    min_len = int(round(min_chunk_sec * sr))
    hop = chunk_len - int(round(overlap_sec * sr))
    if total <= chunk_len:
        return [(0, total)]
    spans, s = [], 0
    while True:
        if s + chunk_len >= total:
            start = s if total - s >= min_len else max(0, total - min_len)
            spans.append((start, total))
            return spans
        spans.append((s, s + chunk_len))
        s += hop


def merge_chunks(spans: Sequence[Tuple[int, int]],
                 read_chunk: Callable[[int], np.ndarray],
                 write: Callable[[np.ndarray], None],
                 crossfade: int) -> int:
    """
    청크를 순서대로 이어 붙인다. 청크 k+1 이 시작하는 지점에서 crossfade 길이만큼
    (이전 청크 -> 다음 청크)로 Hann 형 페이드(합이 1)로 전환한다. 메모리에는 청크 2개만 올라간다.
    반환값: 기록한 총 샘플 수.
    """
    written = 0       # 출력 타임라인 상 확정된 위치
    count = 0
    prev = None
    prev_start = prev_end = 0
    for i, (s, e) in enumerate(spans):
        cur = read_chunk(i)
        if len(cur) != e - s:
            raise PipelineError(ErrorCode.LENGTH_MISMATCH, f"청크 {i} 길이 불일치: {len(cur)} != {e - s}")
        if prev is not None:
            f = min(crossfade, prev_end - s, len(cur))
            pure = prev[written - prev_start: s - prev_start]
            write(pure)
            count += len(pure)
            if f > 0:
                t = (np.arange(f, dtype=np.float32) + 0.5) / f
                ramp = (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
                a = prev[s - prev_start: s - prev_start + f]
                b = cur[:f]
                write(a * (1.0 - ramp) + b * ramp)
                count += f
            written = s + f
        else:
            written = s
        prev, prev_start, prev_end = cur, s, e
    if prev is not None:
        tail = prev[written - prev_start:]
        write(tail)
        count += len(tail)
    return count


# ----------------------------------------------------------------------------
# 5. FFmpeg 래퍼
# ----------------------------------------------------------------------------
@dataclass
class MediaInfo:
    duration: float
    has_audio: bool
    has_video: bool
    audio_rate: int
    audio_channels: int
    format_name: str


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


class FFmpegRunner:
    def __init__(self, cfg: PipelineConfig, cancel: threading.Event):
        self.cfg, self.cancel = cfg, cancel

    def probe(self, path: Path) -> MediaInfo:
        cmd = [self.cfg.ffprobe_path, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60, creationflags=_NO_WINDOW)
        except FileNotFoundError:
            raise PipelineError(ErrorCode.FFMPEG_NOT_FOUND)
        except subprocess.TimeoutExpired:
            raise PipelineError(ErrorCode.UNSUPPORTED_FORMAT, "ffprobe 응답 없음")
        if r.returncode != 0:
            raise PipelineError(ErrorCode.UNSUPPORTED_FORMAT, detail=r.stderr.decode("utf-8", "replace")[-1000:])
        try:
            data = json.loads(r.stdout.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            raise PipelineError(ErrorCode.UNSUPPORTED_FORMAT, "ffprobe 출력 해석 실패")
        streams = data.get("streams", [])
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        has_video = any(s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) != 1
                        for s in streams)
        fmt = data.get("format", {})
        dur = 0.0
        for v in (fmt.get("duration"), (audio or {}).get("duration")):
            try:
                if v is not None and float(v) > 0:
                    dur = float(v)
                    break
            except (TypeError, ValueError):
                pass
        return MediaInfo(
            duration=dur,
            has_audio=audio is not None,
            has_video=has_video,
            audio_rate=int((audio or {}).get("sample_rate") or 0),
            audio_channels=int((audio or {}).get("channels") or 0),
            format_name=fmt.get("format_name", ""),
        )

    def run(self, args: List[str], duration: Optional[float] = None,
            on_progress: Optional[Callable[[float], None]] = None) -> None:
        """ffmpeg 실행. 진행률 콜백, 취소, 무응답/전체 타임아웃, stderr 수집."""
        cmd = [self.cfg.ffmpeg_path, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
               "-progress", "pipe:1", "-nostats", *args]
        err_f = tempfile.TemporaryFile()
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_f, stdin=subprocess.DEVNULL,
                                    text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
        except OSError as error:
            err_f.close()
            if isinstance(error, FileNotFoundError):
                raise PipelineError(ErrorCode.FFMPEG_NOT_FOUND) from error
            raise

        last = {"t": time.monotonic()}

        def reader():
            for line in proc.stdout:  # type: ignore[union-attr]
                last["t"] = time.monotonic()
                line = line.strip()
                if on_progress and duration and line.startswith(("out_time_us=", "out_time_ms=")):
                    try:
                        us = int(line.split("=", 1)[1])
                    except ValueError:
                        continue
                    if us >= 0:
                        on_progress(min(1.0, us / 1e6 / duration))

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        start = time.monotonic()
        try:
            while True:
                try:
                    rc = proc.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.monotonic()
                if self.cancel.is_set():
                    proc.kill()
                    proc.wait()
                    raise PipelineError(ErrorCode.CANCELLED)
                if now - last["t"] > self.cfg.ffmpeg_stall_timeout_sec or \
                        now - start > self.cfg.ffmpeg_total_timeout_sec:
                    proc.kill()
                    proc.wait()
                    raise PipelineError(ErrorCode.FFMPEG_TIMEOUT)
            if rc != 0:
                err_f.seek(0)
                tail = err_f.read().decode("utf-8", "replace")[-2000:]
                raise PipelineError(ErrorCode.FFMPEG_FAILED, detail=tail)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            th.join()
            proc.stdout.close()
            err_f.close()


# ----------------------------------------------------------------------------
# 6. 진행 상황 / 결과
# ----------------------------------------------------------------------------
@dataclass
class Progress:
    stage: str                 # validate | extract | process | merge | mux | done
    stage_fraction: float
    overall: float             # 0.0 ~ 1.0
    done_chunks: int = 0
    total_chunks: int = 0
    eta_sec: Optional[float] = None


@dataclass
class JobResult:
    output_path: Path
    duration_sec: float
    total_chunks: int
    elapsed_sec: float
    warnings: List[str] = field(default_factory=list)


_STAGES = [("validate", 0.02), ("extract", 0.08), ("process", 0.75), ("merge", 0.05), ("mux", 0.10)]
_STAGE_START = {}
_acc = 0.0
for _n, _w in _STAGES:
    _STAGE_START[_n] = (_acc, _w)
    _acc += _w


def _is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return isinstance(e, MemoryError) or "out of memory" in msg or "failed to allocate" in msg


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------------
# 7. 파이프라인 본체
# ----------------------------------------------------------------------------
class AudioProtectionPipeline:
    def __init__(self, processor: ChunkProcessor, config: Optional[PipelineConfig] = None,
                 on_progress: Optional[Callable[[Progress], None]] = None):
        self.processor = processor
        self.cfg = config or PipelineConfig()
        self.on_progress = on_progress
        self._cancel = threading.Event()
        self.ff = FFmpegRunner(self.cfg, self._cancel)
        self._bs = self.cfg.batch_size
        self._done = 0
        self._total = 0

    # ---- 외부에서 호출 ----
    def cancel(self) -> None:
        self._cancel.set()

    def run(self, input_path, output_path=None, overwrite: bool = False) -> JobResult:
        self._cancel.clear()
        self._bs = self.cfg.batch_size
        try:
            return self._run(Path(input_path), output_path, overwrite)
        except PipelineError:
            raise
        except OSError as e:
            code = ErrorCode.LOW_STORAGE if e.errno == errno.ENOSPC else ErrorCode.UNKNOWN
            raise PipelineError(code, str(e)) from e
        except Exception as e:  # 예상 못한 오류도 코드로 감싸서 UI 에 전달
            log.exception("unexpected error")
            raise PipelineError(ErrorCode.UNKNOWN, str(e)) from e

    # ---- 내부 구현 ----
    def _emit(self, stage: str, frac: float, eta: Optional[float] = None) -> None:
        if not self.on_progress:
            return
        if stage == "done":
            p = Progress("done", 1.0, 1.0, self._done, self._total, 0.0)
        else:
            start, w = _STAGE_START[stage]
            frac = max(0.0, min(1.0, frac))
            p = Progress(stage, frac, start + w * frac, self._done, self._total, eta)
        try:
            self.on_progress(p)
        except Exception:  # UI 콜백 오류가 작업을 죽이면 안 됨
            log.exception("progress callback failed")

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise PipelineError(ErrorCode.CANCELLED)

    def _run(self, src: Path, output_path, overwrite: bool) -> JobResult:
        t_start = time.monotonic()
        warnings: List[str] = []
        sr = self.cfg.model_sample_rate

        # 1) 검증
        info = self._validate(src)
        self._emit("validate", 1.0)
        final_out = self._resolve_output(src, output_path, info, overwrite)
        self._check_disk(src, info, final_out)

        # 2) 작업 폴더 (같은 입력+설정이면 같은 폴더 -> 자동 재개)
        job = self._job_dir(src)
        (job / "chunks_out").mkdir(parents=True, exist_ok=True)
        state_path = job / "state.json"
        state = {}
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}

        def save_state(**kw):
            state.update(kw)
            _atomic_write_text(state_path, json.dumps(state))

        # 3) 오디오 추출 -> raw s16le (헤더 없음, 크기/2 = 샘플 수)
        raw = job / "audio.s16"
        if not (state.get("extracted") and raw.exists()):
            tmp = job / "audio.s16.partial"
            self.ff.run(
                ["-i", str(src), "-map", "0:a:0", "-vn",
                 # 오디오 시작이 늦거나 끊긴 경우도 t=0 기준으로 맞춰서 영상과 싱크 유지
                 "-af", "aresample=async=1:first_pts=0",
                 "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", "-f", "s16le", str(tmp)],
                duration=info.duration, on_progress=lambda f: self._emit("extract", f))
            os.replace(tmp, raw)
            save_state(extracted=True)
        total_samples = raw.stat().st_size // 2
        if total_samples == 0:
            raise PipelineError(ErrorCode.NO_AUDIO_STREAM, "추출된 오디오가 비어 있습니다")
        self._emit("extract", 1.0)
        if info.duration and abs(total_samples / sr - info.duration) > 1.5:
            warnings.append(f"추출 오디오 길이({total_samples / sr:.1f}s)가 컨테이너 길이({info.duration:.1f}s)와 다릅니다")

        # 4) 청크 계획 + 처리
        spans = plan_chunks(total_samples, sr, self.cfg.chunk_sec, self.cfg.min_chunk_sec, self.cfg.overlap_sec)
        self._total = len(spans)
        self._process_chunks(raw, job / "chunks_out", spans)

        # 5) 병합 (crossfade)
        self._check_cancel()
        merged = job / "merged.f32"
        crossfade = int(round(self.cfg.crossfade_sec * sr))
        with open(merged, "wb") as fh:
            n = merge_chunks(spans, lambda i: self._load_chunk(job / "chunks_out", i),
                             lambda a: fh.write(np.ascontiguousarray(a, dtype="<f4").tobytes()), crossfade)
        if n != total_samples:
            raise PipelineError(ErrorCode.LENGTH_MISMATCH, f"병합 결과 {n} != 원본 {total_samples} 샘플")
        self._emit("merge", 1.0)

        # 6) 영상 재결합 / 오디오 인코딩 -> 임시 파일에 쓰고 완료 후 이름 변경
        self._check_cancel()
        partial = final_out.with_name(f".{final_out.stem}.partial{final_out.suffix}")
        try:
            self._mux(src, info, merged, partial)
            os.replace(partial, final_out)
        finally:
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass

        # 7) 결과 검증 (길이 비교는 경고만)
        try:
            out_info = self.ff.probe(final_out)
            if info.duration and abs(out_info.duration - info.duration) > 1.0:
                warnings.append(f"출력 길이({out_info.duration:.1f}s)가 원본({info.duration:.1f}s)과 다릅니다")
        except PipelineError:
            warnings.append("출력 파일 검증(ffprobe)에 실패했습니다")

        # 8) 정리
        if self.cfg.cleanup_on_success:
            shutil.rmtree(job, ignore_errors=True)
        self._emit("done", 1.0)
        return JobResult(final_out, info.duration, len(spans), time.monotonic() - t_start, warnings)

    # ---- 1단계: 검증 ----
    def _validate(self, src: Path) -> MediaInfo:
        if not src.is_file():
            raise PipelineError(ErrorCode.FILE_NOT_FOUND)
        ext = src.suffix.lower()
        if ext not in self.cfg.video_exts + self.cfg.audio_exts:
            raise PipelineError(ErrorCode.UNSUPPORTED_FORMAT, f"지원하지 않는 확장자: {ext}")
        if not os.access(src, os.R_OK):
            raise PipelineError(ErrorCode.FILE_NOT_FOUND, "파일을 읽을 수 없습니다")
        size = src.stat().st_size
        if self.cfg.max_file_bytes and size > self.cfg.max_file_bytes:
            raise PipelineError(ErrorCode.TOO_LARGE)
        info = self.ff.probe(src)
        if not info.has_audio:
            raise PipelineError(ErrorCode.NO_AUDIO_STREAM)
        if info.duration <= 0:
            raise PipelineError(ErrorCode.UNSUPPORTED_FORMAT, "길이를 알 수 없습니다")
        if info.duration > self.cfg.max_duration_sec:
            raise PipelineError(ErrorCode.TOO_LONG, f"{info.duration / 60:.0f}분 > 제한 {self.cfg.max_duration_sec / 60:.0f}분")
        return info

    # ---- 출력 경로 ----
    def _resolve_output(self, src: Path, output_path, info: MediaInfo, overwrite: bool) -> Path:
        if output_path:
            out = Path(output_path)
        else:
            out_dir = self.cfg.output_dir or src.parent
            if info.has_video:
                ext = src.suffix.lower() if src.suffix.lower() in (".mp4", ".mov", ".m4v", ".mkv") else ".mkv"
            else:
                ext = src.suffix.lower() if src.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a") else ".flac"
            out = out_dir / f"{src.stem}{self.cfg.output_suffix}{ext}"
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            probe_file = out.parent / f".write_test_{os.getpid()}"
            probe_file.write_bytes(b"")
            probe_file.unlink()
        except OSError as e:
            raise PipelineError(ErrorCode.OUTPUT_PATH_ERROR, str(e)) from e
        if out.resolve() == src.resolve():
            raise PipelineError(ErrorCode.OUTPUT_PATH_ERROR, "원본 파일을 덮어쓸 수 없습니다")
        if out.exists() and not overwrite:   # 원본/기존 결과 보호: 새 이름 부여
            i = 1
            while True:
                cand = out.with_name(f"{out.stem}_{i}{out.suffix}")
                if not cand.exists():
                    out = cand
                    break
                i += 1
        return out

    def _check_disk(self, src: Path, info: MediaInfo, final_out: Path) -> None:
        self.cfg.work_root.mkdir(parents=True, exist_ok=True)
        # 추출 PCM16, 병합 float32, 겹치는 청크 float32를 함께 보관한다.
        overlap_factor = self.cfg.chunk_sec / (self.cfg.chunk_sec - self.cfg.overlap_sec)
        bytes_per_second = self.cfg.model_sample_rate * (2 + 4 + 4 * overlap_factor)
        work_need = int(info.duration * bytes_per_second) + self.cfg.disk_margin_bytes
        out_need = int(src.stat().st_size * 1.1) + self.cfg.disk_margin_bytes
        same_dev = os.stat(self.cfg.work_root).st_dev == os.stat(final_out.parent).st_dev
        checks = [(self.cfg.work_root, work_need + out_need if same_dev else work_need)]
        if not same_dev:
            checks.append((final_out.parent, out_need))
        for d, need in checks:
            if shutil.disk_usage(d).free < need:
                raise PipelineError(ErrorCode.LOW_STORAGE, f"{d}: 필요 약 {need / 1024 ** 3:.1f}GB")

    def _job_dir(self, src: Path) -> Path:
        st = src.stat()
        c = self.cfg
        sig = (f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{c.model_sample_rate}|{c.chunk_sec}|"
               f"{c.min_chunk_sec}|{c.overlap_sec}|{getattr(self.processor, 'tag', 'model')}")
        job = c.work_root / hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
        job.mkdir(parents=True, exist_ok=True)
        return job

    # ---- 청크 입출력 ----
    @staticmethod
    def _read_chunk(raw: Path, start: int, end: int) -> np.ndarray:
        a = np.fromfile(raw, dtype="<i2", count=end - start, offset=start * 2)
        if len(a) != end - start:
            raise PipelineError(ErrorCode.LENGTH_MISMATCH, "추출 오디오를 끝까지 읽지 못했습니다")
        return a.astype(np.float32) / 32768.0

    @staticmethod
    def _chunk_path(d: Path, i: int) -> Path:
        return d / f"chunk_{i:06d}.npy"

    def _save_chunk(self, d: Path, i: int, arr: np.ndarray) -> None:
        final = self._chunk_path(d, i)
        tmp = final.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, arr.astype(np.float32), allow_pickle=False)
        os.replace(tmp, final)   # 원자적 교체: 중간에 죽어도 반쪽 파일이 남지 않음

    def _load_chunk(self, d: Path, i: int) -> np.ndarray:
        p = self._chunk_path(d, i)
        try:
            return np.load(p, allow_pickle=False)
        except Exception as e:
            try:
                p.unlink()    # 다음 실행 때 이 청크만 다시 처리
            except OSError:
                pass
            raise PipelineError(ErrorCode.CHUNK_CORRUPT, str(e)) from e

    # ---- 청크 처리 루프 (배치 / 재시도 / 재개) ----
    def _process_chunks(self, raw: Path, out_dir: Path, spans: List[Tuple[int, int]]) -> None:
        pending = [i for i in range(len(spans)) if not self._chunk_path(out_dir, i).exists()]
        self._done = len(spans) - len(pending)
        self._emit("process", self._done / max(1, len(spans)))
        t0, done_new, pos = time.monotonic(), 0, 0
        while pos < len(pending):
            self._check_cancel()
            idx = pending[pos: pos + self._bs]
            arrays = [self._read_chunk(raw, *spans[i]) for i in idx]
            outs = self._infer(arrays)
            for i, o in zip(idx, outs):
                self._save_chunk(out_dir, i, o)
            pos += len(idx)
            done_new += len(idx)
            self._done += len(idx)
            elapsed = time.monotonic() - t0
            eta = elapsed / done_new * (len(pending) - pos) if done_new else None
            self._emit("process", self._done / len(spans), eta)

    def _infer(self, arrays: List[np.ndarray]) -> List[np.ndarray]:
        """모델 호출. OOM -> 배치 절반으로 분할, 그 외 오류 -> 백오프 재시도 후 MODEL_ERROR."""
        # 현재 배치 크기(OOM 으로 줄었을 수 있음)보다 크면 먼저 쪼갠다
        if len(arrays) > self._bs:
            bs = self._bs          # 루프 중 _bs 가 바뀌어도 슬라이스 간격은 고정 (청크 누락 방지)
            outs_all: List[np.ndarray] = []
            for k in range(0, len(arrays), bs):
                outs_all += self._infer(arrays[k: k + bs])
            if len(outs_all) != len(arrays):
                raise PipelineError(ErrorCode.MODEL_ERROR, "배치 분할 후 출력 개수 불일치")
            return outs_all
        attempt = 0
        while True:
            self._check_cancel()
            try:
                outs = self.processor.process_batch(arrays)
                return self._validate_outputs(arrays, outs)
            except PipelineError:
                raise
            except Exception as e:
                if _is_oom(e) and len(arrays) > 1:
                    self._bs = max(1, len(arrays) // 2)
                    log.warning("OOM: 배치 크기를 %d 로 줄입니다", self._bs)
                    return self._infer(arrays)
                attempt += 1
                log.warning("모델 호출 실패 (%d/%d): %s", attempt, self.cfg.max_retries, e)
                if attempt > self.cfg.max_retries:
                    raise PipelineError(ErrorCode.MODEL_ERROR, detail=repr(e)) from e
                if self._cancel.wait(self.cfg.retry_backoff_sec * attempt):
                    raise PipelineError(ErrorCode.CANCELLED)

    @staticmethod
    def _validate_outputs(ins: List[np.ndarray], outs) -> List[np.ndarray]:
        if len(outs) != len(ins):
            raise ValueError(f"출력 개수 불일치: {len(outs)} != {len(ins)}")
        result = []
        for a, o in zip(ins, outs):
            o = np.asarray(o, dtype=np.float32).reshape(-1)
            if len(o) != len(a):
                raise ValueError(f"출력 길이 불일치: {len(o)} != {len(a)}")
            if not np.all(np.isfinite(o)):
                raise ValueError("출력에 NaN/Inf 가 있습니다")
            result.append(np.clip(o, -1.0, 1.0))
        return result

    # ---- 재결합 ----
    def _mux(self, src: Path, info: MediaInfo, merged: Path, out: Path) -> None:
        sr = self.cfg.model_sample_rate
        raw_in = ["-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", str(merged)]
        ext = out.suffix.lower()
        restore: List[str] = []
        if self.cfg.restore_source_format and info.audio_rate > 0:
            restore = ["-ar", str(info.audio_rate), "-ac", str(max(1, min(2, info.audio_channels)))]

        if info.has_video:
            codec = ["-c:a", "aac", "-b:a", self.cfg.audio_bitrate]
            args = ["-i", str(src), *raw_in, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    *codec, *restore, "-map_metadata", "0"]
            if ext in (".mp4", ".mov", ".m4v"):
                args += ["-movflags", "+faststart"]
        else:
            codec = {".wav": ["-c:a", "pcm_s24le"], ".flac": ["-c:a", "flac"],
                     ".mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
                     ".m4a": ["-c:a", "aac", "-b:a", self.cfg.audio_bitrate],
                     ".mp4": ["-c:a", "aac", "-b:a", self.cfg.audio_bitrate]}.get(ext, ["-c:a", "flac"])
            args = [*raw_in, *codec, *restore]
        self.ff.run([*args, str(out)], duration=info.duration, on_progress=lambda f: self._emit("mux", f))
        self._emit("mux", 1.0)


# ----------------------------------------------------------------------------
# 8. 여러 파일 배치 처리
# ----------------------------------------------------------------------------
@dataclass
class FileResult:
    path: Path
    ok: bool
    output_path: Optional[Path] = None
    error_code: Optional[ErrorCode] = None
    message: str = ""


def process_files(paths: Sequence, pipeline: AudioProtectionPipeline,
                  on_file_start: Optional[Callable[[int, int, Path], None]] = None) -> List[FileResult]:
    """파일을 순서대로 처리. 한 파일이 실패해도 다음 파일 계속 진행. 취소되면 남은 파일은 중단."""
    results: List[FileResult] = []
    for n, p in enumerate(paths):
        p = Path(p)
        if on_file_start:
            on_file_start(n, len(paths), p)
        try:
            r = pipeline.run(p)
            results.append(FileResult(p, True, r.output_path))
        except PipelineError as e:
            results.append(FileResult(p, False, None, e.code, e.user_message))
            if e.code == ErrorCode.CANCELLED:
                break
    return results


def cleanup_old_jobs(work_root: Path, max_age_days: float = 3.0) -> int:
    """앱 시작 시 호출: 오래된(재개 가능성 없는) 작업 폴더 삭제. 삭제한 폴더 수 반환."""
    removed, now = 0, time.time()
    if not Path(work_root).is_dir():
        return 0
    for d in Path(work_root).iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age_days * 86400:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            pass
    return removed


# ----------------------------------------------------------------------------
# 9. 간단 CLI (더미 모델로 전체 흐름 확인):  python audio_pipeline.py input.mp4
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 2:
        print("usage: python audio_pipeline.py <input> [output]")
        sys.exit(1)

    def show(p: Progress):
        eta = f" ETA {p.eta_sec:.0f}s" if p.eta_sec else ""
        print(f"\r[{p.stage:8s}] {p.overall * 100:5.1f}%  chunks {p.done_chunks}/{p.total_chunks}{eta}   ",
              end="", flush=True)

    pipe = AudioProtectionPipeline(DummyProcessor(), PipelineConfig(), on_progress=show)
    try:
        res = pipe.run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print(f"\n완료: {res.output_path}  ({res.elapsed_sec:.1f}s, 청크 {res.total_chunks}개)")
        for w in res.warnings:
            print("경고:", w)
    except PipelineError as e:
        print(f"\n실패: {e.code.value} - {e.user_message}\n{e.detail}")
        sys.exit(2)
