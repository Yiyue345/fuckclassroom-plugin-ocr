from __future__ import annotations

import base64
import hashlib
import io
import mimetypes
import re
import shutil
import threading
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from .compat import install_rapidocr_compat
from .errors import OcrError
from .engine import (
    increment_ocr_stat,
    rapidocr_ocr,
    tesseract_ocr,
    write_cache_text,
)
from fuckclassroom.core.plugins import PluginContext
from fuckclassroom.plugins.process_runtime import ProcessPluginError, ProcessPluginHost
from fuckclassroom.plugins.rpc import PLUGIN_RPC_API_VERSION


_RPC_IMAGE_TARGET_BYTES = 2_500_000
_RPC_IMAGE_MIN_SIDE = 768
_RPC_JPEG_QUALITIES = (90, 82, 74, 66)


def _prepare_rpc_image(
    image_bytes: bytes,
    image_name: str,
    *,
    max_side: int,
) -> tuple[bytes, str]:
    """Keep image RPC payloads safely below the 4 MiB JSON message limit."""
    if len(image_bytes) <= _RPC_IMAGE_TARGET_BYTES:
        return image_bytes, image_name

    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            image = source.copy()

        if "A" in image.getbands():
            background = Image.new("RGB", image.size, "white")
            background.paste(image, mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        bounded_side = max(_RPC_IMAGE_MIN_SIDE, int(max_side))
        if max(image.size) > bounded_side:
            image.thumbnail((bounded_side, bounded_side), resampling)

        prepared_name = f"{Path(image_name).stem or 'image'}.jpg"
        for quality in _RPC_JPEG_QUALITIES:
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            prepared = output.getvalue()
            if len(prepared) <= _RPC_IMAGE_TARGET_BYTES:
                return prepared, prepared_name

        current = image
        while max(current.size) > _RPC_IMAGE_MIN_SIDE:
            width, height = current.size
            scale = max(_RPC_IMAGE_MIN_SIDE / max(width, height), 0.82)
            next_size = (
                max(1, int(width * scale)),
                max(1, int(height * scale)),
            )
            if next_size == current.size:
                break
            current = current.resize(next_size, resampling)
            output = io.BytesIO()
            current.save(output, format="JPEG", quality=74, optimize=True)
            prepared = output.getvalue()
            if len(prepared) <= _RPC_IMAGE_TARGET_BYTES:
                return prepared, prepared_name
    except Exception as exc:
        raise OcrError(
            "OCR 图片超过 RPC 安全上限，且无法完成缩放/压缩"
        ) from exc

    raise OcrError(
        "OCR 图片压缩后仍超过 RPC 安全上限，请降低 OCR 图片最大边设置"
    )


class OcrService:
    def __init__(self, config, *, ai_ocr=None, ai_available=None) -> None:
        self.config = config
        self._ai_ocr = ai_ocr
        self._ai_available = ai_available

    def ocr_image(
        self,
        image_bytes: bytes,
        image_name: str,
        *,
        stats: dict[str, int] | None = None,
        stats_lock: threading.Lock | None = None,
    ) -> str:
        engine = self.config.ocr_engine
        if engine in ("auto", "rapidocr"):
            try:
                cache_path = self._rapidocr_cache_path(image_bytes)
                if cache_path is not None and cache_path.exists():
                    try:
                        text = cache_path.read_text(encoding="utf-8")
                    except OSError:
                        pass
                    else:
                        increment_ocr_stat(stats, stats_lock, "cache_hits")
                        return text

                text = rapidocr_ocr(
                    image_bytes,
                    onnx_threads=self.config.ocr_onnx_threads,
                    max_side=self.config.ocr_max_side,
                    use_angle_cls=self.config.ocr_use_angle_cls,
                )
                increment_ocr_stat(stats, stats_lock, "inferences")
                if cache_path is not None:
                    write_cache_text(cache_path, text)
                return text
            except ImportError:
                if engine == "rapidocr":
                    raise OcrError(
                        "缺少 rapidocr-onnxruntime，请先安装项目依赖"
                    ) from None

        if engine in ("auto", "tesseract") and shutil.which("tesseract"):
            increment_ocr_stat(stats, stats_lock, "inferences")
            return tesseract_ocr(image_bytes, image_name)

        if engine in ("auto", "ai"):
            ai_available = bool(
                self.config.ai_api_key
                and self._ai_ocr is not None
                and (
                    self._ai_available is None
                    or bool(self._ai_available())
                )
            )
            if ai_available:
                media_type = mimetypes.guess_type(image_name)[0] or "image/png"
                increment_ocr_stat(stats, stats_lock, "inferences")
                return str(self._ai_ocr(image_bytes, media_type))
            if engine == "ai":
                raise OcrError("AI OCR 需要启用 AI 总结插件并配置 AI API Key")

        raise OcrError("未找到可用 OCR 引擎。")

    def can_ocr(self, warnings: list[str]) -> bool:
        engine = self.config.ocr_engine
        if engine in ("auto", "rapidocr"):
            try:
                import rapidocr_onnxruntime  # noqa: F401

                return True
            except ImportError:
                if engine == "rapidocr":
                    warnings.append("已跳过 OCR：缺少 rapidocr-onnxruntime。")
                    return False
        if engine in ("auto", "tesseract") and shutil.which("tesseract"):
            return True
        if engine in ("auto", "ai") and self.config.ai_api_key:
            if self._ai_ocr is not None and (
                self._ai_available is None or bool(self._ai_available())
            ):
                return True
            if engine == "ai":
                warnings.append("已跳过 OCR：AI 总结插件未启用。")
                return False
        warnings.append("已跳过 OCR：未找到可用 OCR 引擎。")
        return False

    def _rapidocr_cache_path(self, image_bytes: bytes) -> Path | None:
        if not self.config.ocr_cache_enabled:
            return None
        digest = hashlib.sha256(image_bytes).hexdigest()
        try:
            rapidocr_version = package_version("rapidocr-onnxruntime")
        except PackageNotFoundError:
            rapidocr_version = "unknown"
        normalized_version = re.sub(r"[^A-Za-z0-9._-]+", "_", rapidocr_version)
        namespace = (
            f"rapidocr-{normalized_version}-side{self.config.ocr_max_side}-"
            f"cls{int(self.config.ocr_use_angle_cls)}"
        )
        return (
            self.config.cache_dir
            / "ocr"
            / namespace
            / digest[:2]
            / f"{digest}.txt"
        )


class OcrProcessProxy:
    def __init__(self, host: ProcessPluginHost, *, max_side: int = 1600) -> None:
        self.host = host
        self.max_side = max_side

    def ocr_image(
        self,
        image_bytes: bytes,
        image_name: str,
        *,
        stats: dict[str, int] | None = None,
        stats_lock: threading.Lock | None = None,
    ) -> str:
        rpc_image_bytes, rpc_image_name = _prepare_rpc_image(
            image_bytes,
            image_name,
            max_side=self.max_side,
        )
        try:
            payload = self.host.call_sync(
                "ocr.image",
                {
                    "image_b64": base64.b64encode(rpc_image_bytes).decode("ascii"),
                    "image_name": rpc_image_name,
                },
                timeout=300,
            )
        except ProcessPluginError as exc:
            raise OcrError(str(exc)) from exc
        if not isinstance(payload, dict):
            raise OcrError("OCR 子进程返回格式错误")
        deltas = payload.get("stats")
        if isinstance(deltas, dict) and stats is not None:
            if stats_lock is None:
                for key, value in deltas.items():
                    stats[str(key)] = stats.get(str(key), 0) + int(value)
            else:
                with stats_lock:
                    for key, value in deltas.items():
                        stats[str(key)] = stats.get(str(key), 0) + int(value)
        return str(payload.get("text") or "")

    def can_ocr(self, warnings: list[str]) -> bool:
        try:
            payload = self.host.call_sync("ocr.can_ocr", {}, timeout=30)
        except ProcessPluginError as exc:
            raise OcrError(str(exc)) from exc
        if not isinstance(payload, dict):
            raise OcrError("OCR 子进程返回格式错误")
        child_warnings = payload.get("warnings")
        if isinstance(child_warnings, list):
            warnings.extend(str(item) for item in child_warnings)
        return bool(payload.get("available"))


def setup_services(context: PluginContext) -> None:
    host = ProcessPluginHost(
        plugin_id="ocr",
        root=Path(__file__).resolve().parent,
        entry="worker.py",
        data_dir=Path(context.config.data_dir),
        rpc_registry=context.services.get("plugin_rpc"),
        rpc_api_version=PLUGIN_RPC_API_VERSION,
        rpc_permissions=("core.plugins.has", "ai_summary.ocr.image"),
    )
    context.services.add("ocr_process_host", host)
    context.services.add(
        "ocr_service",
        OcrProcessProxy(host, max_side=context.config.ocr_max_side),
    )


async def startup(context: PluginContext) -> None:
    await context.services.get("ocr_process_host").start()


async def shutdown(context: PluginContext) -> None:
    await context.services.get("ocr_process_host").stop()


__all__ = [
    "OcrError",
    "OcrProcessProxy",
    "OcrService",
    "setup_services",
    "shutdown",
    "startup",
]
