from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from fuckclassroom.core.config import AppConfig

from .compat import install_rapidocr_compat
from .services import OcrService, _LOCAL_IMAGE_MAX_BYTES, _prepare_rpc_image


_service: OcrService | None = None


def _get_service(context) -> OcrService:
    global _service
    if _service is None:
        install_rapidocr_compat()

        def ai_available() -> bool:
            return bool(
                context.rpc.call(
                    "core.plugins.has",
                    {"plugin_id": "ai_summary"},
                )
            )

        def ai_ocr(image_bytes: bytes, media_type: str) -> str:
            # Only the AI cross-plugin RPC uses the 4 MiB JSON limit.
            # Local RapidOCR/Tesseract must see unmodified input bytes.
            extension = mimetypes.guess_extension(media_type) or ".png"
            bounded, name = _prepare_rpc_image(
                image_bytes, "image" + extension,
                max_side=context.config.ocr_max_side if hasattr(context, "config") else
                    AppConfig(data_dir=context.data_dir).ocr_max_side,
            )
            payload = context.rpc.call(
                "ai_summary.ocr.image",
                {
                    "image_b64": base64.b64encode(bounded).decode("ascii"),
                    "media_type": mimetypes.guess_type(name)[0] or media_type,
                },
            )
            return str(payload or "")

        _service = OcrService(
            AppConfig(data_dir=context.data_dir),
            ai_ocr=ai_ocr,
            ai_available=ai_available,
        )
    return _service


def handle_call(method, params, context, progress):
    service = _get_service(context)
    if method == "ocr.can_ocr":
        warnings: list[str] = []
        return {
            "available": service.can_ocr(warnings),
            "warnings": warnings,
        }
    if method == "ocr.image":
        encoded = str(params.get("image_b64") or "")
        image_name = str(params.get("image_name") or "image.png")
        raw_path = str(params.get("image_path") or "")
        if raw_path:
            root = (Path(context.data_dir) / "ocr_rpc_inputs").resolve()
            image_path = Path(raw_path)
            if (not image_path.is_file() or image_path.is_symlink()
                    or not image_path.resolve().is_relative_to(root)):
                raise ValueError("OCR 临时文件路径非法")
            if image_path.stat().st_size > _LOCAL_IMAGE_MAX_BYTES:
                raise ValueError("OCR 临时文件超过 100 MB")
            image_bytes = image_path.read_bytes()
        else:
            image_bytes = base64.b64decode(encoded, validate=True)
        stats = {"cache_hits": 0, "inferences": 0}
        text = service.ocr_image(
            image_bytes,
            image_name,
            stats=stats,
        )
        return {"text": text, "stats": stats}
    raise ValueError(f"未知 OCR Worker 方法：{method}")
