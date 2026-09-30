from __future__ import annotations

import base64

from fuckclassroom.core.config import AppConfig

from .compat import install_rapidocr_compat
from .services import OcrService


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
            payload = context.rpc.call(
                "ai_summary.ocr.image",
                {
                    "image_b64": base64.b64encode(image_bytes).decode("ascii"),
                    "media_type": media_type,
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
        image_bytes = base64.b64decode(encoded, validate=False)
        stats = {"cache_hits": 0, "inferences": 0}
        text = service.ocr_image(
            image_bytes,
            image_name,
            stats=stats,
        )
        return {"text": text, "stats": stats}
    raise ValueError(f"未知 OCR Worker 方法：{method}")
