"""Availability checks must tell users why PPT OCR is skipped."""

from __future__ import annotations

import ast
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def service(engine: str, *, key: str = "", ai_ocr=None, ai_available=None):
    path = ROOT / "services.py"
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "OcrService")
    code = ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        cls,
    ], type_ignores=[]))
    namespace = {"shutil": shutil}
    exec(compile(code, str(path), "exec"), namespace)
    return namespace["OcrService"](
        SimpleNamespace(ocr_engine=engine, ai_api_key=key),
        ai_ocr=ai_ocr,
        ai_available=ai_available,
    )


class OcrAvailabilityTests(unittest.TestCase):
    def test_rapidocr_dependency_missing_has_actionable_message(self):
        messages = []
        with patch.dict(sys.modules, {"rapidocr_onnxruntime": None}):
            self.assertFalse(service("rapidocr").can_ocr(messages))
        self.assertIn("rapidocr-onnxruntime", messages[0])
        self.assertIn("设置", messages[0])

    def test_tesseract_missing_in_path_has_actionable_message(self):
        messages = []
        with patch.object(shutil, "which", return_value=None):
            self.assertFalse(service("tesseract").can_ocr(messages))
        self.assertIn("PATH", messages[0])
        self.assertIn("Tesseract", messages[0])

    def test_ai_without_key_reports_missing_api_key(self):
        messages = []
        self.assertFalse(service("ai").can_ocr(messages))
        self.assertIn("AI API Key", messages[0])
        self.assertIn("AI 总结插件", messages[0])

    def test_ai_service_disabled_is_distinct_from_key_absence(self):
        messages = []
        self.assertFalse(service("ai", key="saved", ai_ocr=lambda *a: "x", ai_available=lambda: False).can_ocr(messages))
        self.assertIn("未启用", messages[0])
        self.assertNotIn("尚未配置", messages[0])

    def test_auto_missing_all_backends_explains_next_steps(self):
        messages = []
        with patch.dict(sys.modules, {"rapidocr_onnxruntime": None}), patch.object(shutil, "which", return_value=None):
            self.assertFalse(service("auto").can_ocr(messages))
        self.assertIn("自动 OCR", messages[0])
        self.assertIn("RapidOCR", messages[0])
        self.assertIn("AI API Key", messages[0])

    def test_auto_prefers_healthy_tesseract(self):
        messages = []
        with patch.dict(sys.modules, {"rapidocr_onnxruntime": None}), patch.object(shutil, "which", return_value="/usr/bin/tesseract"):
            self.assertTrue(service("auto").can_ocr(messages))
        self.assertEqual(messages, [])

    def test_ai_with_valid_integration_is_available(self):
        messages = []
        backend = service("ai", key="saved", ai_ocr=lambda *a: "ok", ai_available=lambda: True)
        self.assertTrue(backend.can_ocr(messages))
        self.assertEqual(messages, [])


if __name__ == "__main__":
    unittest.main()
