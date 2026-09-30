from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path


_RAPID_OCR_LOCAL = threading.local()


def rapidocr_ocr(
    image_bytes: bytes,
    *,
    onnx_threads: int = 1,
    max_side: int = 1600,
    use_angle_cls: bool = False,
) -> str:
    threads = max(1, int(onnx_threads))
    side = max(512, int(max_side))
    engine_key = (threads, side)
    engine = getattr(_RAPID_OCR_LOCAL, "engine", None)
    current_key = getattr(_RAPID_OCR_LOCAL, "engine_key", None)
    if engine is None or current_key != engine_key:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR(
            max_side_len=side,
            det_limit_side_len=side,
            det_limit_type="max",
            intra_op_num_threads=threads,
            inter_op_num_threads=1,
            print_verbose=False,
        )
        _RAPID_OCR_LOCAL.engine = engine
        _RAPID_OCR_LOCAL.engine_key = engine_key

    result, _ = engine(image_bytes, use_cls=bool(use_angle_cls))
    if not result:
        return ""
    lines: list[str] = []
    for item in result:
        if len(item) < 2:
            continue
        text = str(item[1]).strip()
        confidence = float(item[2]) if len(item) > 2 else 0.0
        if text and confidence >= 0.35:
            lines.append(text)
    return "\n".join(lines)


def write_cache_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{threading.get_ident()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def increment_ocr_stat(
    stats: dict[str, int] | None,
    lock: threading.Lock | None,
    key: str,
) -> None:
    if stats is None:
        return
    if lock is None:
        stats[key] = stats.get(key, 0) + 1
        return
    with lock:
        stats[key] = stats.get(key, 0) + 1


def tesseract_ocr(image_bytes: bytes, image_name: str) -> str:
    suffix = Path(image_name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as image_file:
        image_file.write(image_bytes)
        image_path = Path(image_file.name)
    try:
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "-l", "chi_sim+eng"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        return result.stdout.strip()
    finally:
        image_path.unlink(missing_ok=True)


# Compatibility aliases for the names used before OCR became its own plugin.
_rapidocr_ocr = rapidocr_ocr
_write_cache_text = write_cache_text
_increment_ocr_stat = increment_ocr_stat
_tesseract_ocr = tesseract_ocr


__all__ = [
    "increment_ocr_stat",
    "rapidocr_ocr",
    "tesseract_ocr",
    "write_cache_text",
]
