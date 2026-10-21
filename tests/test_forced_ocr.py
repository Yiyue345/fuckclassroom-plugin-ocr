"""Regression: a manual PPT OCR rerun must bypass prior RapidOCR text cache."""

from __future__ import annotations

import ast
import hashlib
import re
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def actual_ocr_service(env: dict):
    path = ROOT / "services.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    service = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "OcrService")
    ast_module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        service,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(ast_module), str(path), "exec"), env)
    return env["OcrService"]


class ForcedOcrTests(unittest.TestCase):
    def test_forced_recognition_does_not_reuse_cached_result(self):
        calls = []
        stats = {}
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                ocr_engine="rapidocr",
                ocr_cache_enabled=True,
                ocr_onnx_threads=1,
                ocr_max_side=1600,
                ocr_use_angle_cls=False,
                cache_dir=Path(directory),
            )
            env = {
                "Path": Path,
                "hashlib": hashlib,
                "re": re,
                "package_version": lambda _: "test",
                "PackageNotFoundError": RuntimeError,
                "rapidocr_ocr": lambda data, **kwargs: (
                    calls.append(bytes(data)) or "新识别文本"
                ),
                "increment_ocr_stat": lambda d, lock, key: d.__setitem__(key, d.get(key, 0) + 1),
                "write_cache_text": lambda path, text: (
                    path.parent.mkdir(parents=True, exist_ok=True),
                    path.write_text(text, encoding="utf-8"),
                ),
            }
            service = actual_ocr_service(env)(config)
            image = b"same-image-bytes"
            cached = service._rapidocr_cache_path(image)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text("之前的模型识别", encoding="utf-8")

            regular = service.ocr_image(image, "slide.png", stats=stats)
            self.assertEqual(regular, "之前的模型识别")
            self.assertEqual(calls, [])
            self.assertEqual(stats["cache_hits"], 1)

            regenerated = service.ocr_image(image, "slide.png", stats=stats, force=True)
            self.assertEqual(regenerated, "新识别文本")
            self.assertEqual(calls, [image])
            self.assertEqual(stats["inferences"], 1)
            self.assertEqual(cached.read_text(encoding="utf-8"), "新识别文本")

    def test_proxy_forwards_force_and_worker_reads_it(self):
        proxy = (ROOT / "services.py").read_text(encoding="utf-8")
        worker = (ROOT / "worker.py").read_text(encoding="utf-8")
        self.assertIn('"force": bool(force)', proxy)
        self.assertIn('force=params.get("force") is True', worker)


if __name__ == "__main__":
    unittest.main()
