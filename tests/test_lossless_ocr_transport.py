"""OCR subprocess transport regression tests (host-runtime independent).

Execute the actual transport class and worker functions in isolation, so this
plugin can validate its IPC contract without installing the whole host app.
"""

from __future__ import annotations

import ast
import base64
import io
import mimetypes
import random
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TARGET_BYTES = 2_500_000
MAX_BYTES = 100 * 1024 * 1024


class OcrError(Exception):
    pass


class ProcessPluginError(Exception):
    pass


def _load_actual_definitions(file_name: str, *names: str, env: dict):
    source = ROOT / file_name
    parsed = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    kept = [node for node in parsed.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    kept.extend(
        node
        for node in parsed.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    )
    if len(kept) != len(names) + 1:
        raise AssertionError(f"Missing executable definitions in {file_name}: {names}")
    module = ast.fix_missing_locations(ast.Module(body=kept, type_ignores=[]))
    exec(compile(module, str(source), "exec"), env)
    return env


def _services():
    env = {
        "base64": base64,
        "io": io,
        "Path": Path,
        "tempfile": tempfile,
        "threading": threading,
        "OcrError": OcrError,
        "ProcessPluginError": ProcessPluginError,
        "_RPC_IMAGE_TARGET_BYTES": TARGET_BYTES,
        "_LOCAL_IMAGE_MAX_BYTES": MAX_BYTES,
        "_RPC_IMAGE_MIN_SIDE": 768,
        "_RPC_JPEG_QUALITIES": (90, 82, 74, 66),
    }
    return _load_actual_definitions(
        "services.py", "_prepare_rpc_image", "OcrProcessProxy", env=env
    )


def _worker(services):
    env = {
        "base64": base64,
        "mimetypes": mimetypes,
        "Path": Path,
        "_LOCAL_IMAGE_MAX_BYTES": MAX_BYTES,
        "_prepare_rpc_image": services["_prepare_rpc_image"],
        "AppConfig": lambda data_dir: SimpleNamespace(ocr_max_side=1600),
        "install_rapidocr_compat": lambda: None,
        "_service": None,
    }
    return _load_actual_definitions(
        "worker.py", "_get_service", "handle_call", env=env
    )


def _large_png() -> bytes:
    # Deterministic, fine-grained high-resolution pixels prevent PNG deflation
    # from disguising lossy transformations by shrinking below the IPC limit.
    image = Image.frombytes(
        "L", (1900, 1800), random.Random(83).randbytes(1900 * 1800)
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    data = output.getvalue()
    if len(data) <= TARGET_BYTES:
        raise AssertionError("Test image did not exceed IPC payload limit")
    return data


class LosslessWorkerIpcTests(unittest.TestCase):
    def test_high_resolution_png_reaches_local_worker_byte_for_byte(self):
        services = _services()
        worker = _worker(services)
        original = _large_png()
        self.assertGreater(len(original), TARGET_BYTES)
        seen = {}

        class FakeOcrService:
            def ocr_image(self, image_bytes, image_name, *, stats, force=False):
                seen["image"] = image_bytes
                seen["name"] = image_name
                stats["inferences"] += 1
                return "small-text / table-lines"

        worker["_get_service"] = lambda context: FakeOcrService()

        with tempfile.TemporaryDirectory() as temp:
            context = SimpleNamespace(data_dir=Path(temp))

            class LocalHost:
                data_dir = Path(temp)

                def call_sync(self, method, params, *, timeout):
                    self_args = params
                    self.assertTrue(method == "ocr.image")
                    seen["params"] = self_args
                    return worker["handle_call"](method, params, context, None)

            # Bound self in nested class is LocalHost; make assertions outside.
            def send(self, method, params, *, timeout):
                seen["params"] = dict(params)
                return worker["handle_call"](method, params, context, None)

            LocalHost.call_sync = send
            result = services["OcrProcessProxy"](LocalHost()).ocr_image(
                original, "hires-screenshot.png"
            )
            self.assertEqual(result, "small-text / table-lines")
            self.assertEqual(seen["image"], original)
            self.assertEqual(seen["name"], "hires-screenshot.png")
            self.assertIn("image_path", seen["params"])
            self.assertNotIn("image_b64", seen["params"])
            self.assertFalse(Path(seen["params"]["image_path"]).exists())
            self.assertFalse(list((Path(temp) / "ocr_rpc_inputs").glob("*")))

    def test_small_image_stays_in_base64_worker_path(self):
        services = _services()
        worker = _worker(services)
        seen = {}
        worker["_get_service"] = lambda context: SimpleNamespace(
            ocr_image=lambda payload, image_name, *, stats, force=False: (
                seen.update(image=payload, name=image_name) or "OK"
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            context = SimpleNamespace(data_dir=Path(temp))

            class Host:
                data_dir = Path(temp)

                def call_sync(self, method, params, *, timeout):
                    seen["params"] = dict(params)
                    return worker["handle_call"](method, params, context, None)

            result = services["OcrProcessProxy"](Host()).ocr_image(
                b"small opaque png", "small.png"
            )
            self.assertEqual(result, "OK")
            self.assertEqual(seen["image"], b"small opaque png")
            self.assertEqual(
                base64.b64decode(seen["params"]["image_b64"]), b"small opaque png"
            )
            self.assertNotIn("image_path", seen["params"])

    def test_worker_rejects_outside_temp_directory(self):
        services = _services()
        worker = _worker(services)
        worker["_get_service"] = lambda context: SimpleNamespace(ocr_image=lambda *a, **k: "BAD")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            outside = root / "outside.png"
            outside.write_bytes(b"not allowed")
            context = SimpleNamespace(data_dir=root)
            with self.assertRaisesRegex(ValueError, "路径非法"):
                worker["handle_call"](
                    "ocr.image", {"image_path": str(outside), "image_name": "x.png"},
                    context, None,
                )

    def test_temporary_file_is_cleaned_when_worker_fails(self):
        services = _services()
        original = _large_png()
        with tempfile.TemporaryDirectory() as temp:
            class FailingHost:
                data_dir = Path(temp)

                def call_sync(self, method, params, *, timeout):
                    self.path = params["image_path"]
                    raise ProcessPluginError("worker stopped")

            host = FailingHost()
            with self.assertRaises(OcrError):
                services["OcrProcessProxy"](host).ocr_image(original, "bad.png")
            self.assertFalse(Path(host.path).exists())


class AiOnlyCompressionTests(unittest.TestCase):
    def test_ai_rpc_compresses_large_images_but_local_worker_does_not(self):
        services = _services()
        worker = _worker(services)
        with tempfile.TemporaryDirectory() as temp:
            calls = []

            class FakeOcrService:
                def __init__(self, config, *, ai_ocr, ai_available):
                    self.ai_ocr = ai_ocr
                    self.ai_available = ai_available

            worker["OcrService"] = FakeOcrService
            worker["_service"] = None

            class Rpc:
                def call(self, name, params):
                    calls.append((name, dict(params)))
                    return True if name == "core.plugins.has" else "AI text"

            ctx = SimpleNamespace(data_dir=Path(temp), rpc=Rpc())
            fake_service = worker["_get_service"](ctx)
            self.assertTrue(fake_service.ai_available())

            image = Image.new("RGB", (1200, 1000), "white")
            out = io.BytesIO()
            image.save(out, format="BMP")
            original = out.getvalue()
            self.assertGreater(len(original), TARGET_BYTES)

            self.assertEqual(fake_service.ai_ocr(original, "image/bmp"), "AI text")
            name, payload = calls[-1]
            self.assertEqual(name, "ai_summary.ocr.image")
            compressed = base64.b64decode(payload["image_b64"], validate=True)
            self.assertLessEqual(len(compressed), TARGET_BYTES)
            self.assertNotEqual(compressed, original)
            self.assertEqual(payload["media_type"], "image/jpeg")


if __name__ == "__main__":
    unittest.main()
