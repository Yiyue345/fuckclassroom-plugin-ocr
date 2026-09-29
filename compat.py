from __future__ import annotations

from typing import Any


def install_rapidocr_compat() -> None:
    """Patch rapidocr-onnxruntime 1.x for the parameters used by this project.

    rapidocr-onnxruntime 1.4.x has a fragile det_* parameter remapping path that can
    raise KeyError('model_path') on some packaged builds. We avoid that path by
    preprocessing oversized byte images ourselves and only forwarding the ONNX
    thread controls that the legacy engine handles reliably.
    """
    try:
        import rapidocr_onnxruntime
    except ImportError:
        return

    original = getattr(rapidocr_onnxruntime, "RapidOCR", None)
    if original is None or getattr(original, "_fuckclassroom_compat", False):
        return

    class CompatibleRapidOCR(original):
        _fuckclassroom_compat = True

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._fuckclassroom_max_side = max(
                512,
                int(kwargs.pop("max_side_len", 2000) or 2000),
            )

            # Do not forward these legacy det_* overrides. In rapidocr-onnxruntime
            # 1.4.x they can enter UpdateParameters.update_params() and fail while
            # trying to recover the bundled model_path. The equivalent size limit
            # is applied before inference in __call__ instead.
            kwargs.pop("det_limit_side_len", None)
            kwargs.pop("det_limit_type", None)

            # Keep the settings that matter for CPU oversubscription. These are
            # global parameters in rapidocr-onnxruntime 1.x and are copied into the
            # Det/Cls/Rec ONNX Runtime sessions by the package itself.
            threads = max(1, int(kwargs.get("intra_op_num_threads", 1) or 1))
            kwargs["intra_op_num_threads"] = threads
            kwargs["inter_op_num_threads"] = max(
                1,
                int(kwargs.get("inter_op_num_threads", 1) or 1),
            )
            kwargs.setdefault("print_verbose", False)
            super().__init__(*args, **kwargs)

        def __call__(self, img_content: Any, *args: Any, **kwargs: Any):
            prepared = _resize_byte_image(img_content, self._fuckclassroom_max_side)
            return super().__call__(prepared, *args, **kwargs)

    CompatibleRapidOCR.__name__ = getattr(original, "__name__", "RapidOCR")
    CompatibleRapidOCR.__qualname__ = getattr(original, "__qualname__", "RapidOCR")
    CompatibleRapidOCR.__module__ = getattr(original, "__module__", "rapidocr_onnxruntime")
    rapidocr_onnxruntime.RapidOCR = CompatibleRapidOCR


def _resize_byte_image(value: Any, max_side: int):
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return value

    try:
        import cv2
        import numpy as np
    except ImportError:
        return value

    encoded = np.frombuffer(bytes(value), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim < 2:
        return value

    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )
