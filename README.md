# FuckClassroom PPT OCR 插件

FuckClassroom 的独立 PPT OCR 插件，插件 ID 为 `ocr`。

## 功能

- RapidOCR / ONNX Runtime OCR
- Tesseract OCR 回退
- OCR 结果缓存
- AI OCR 可选回退
- 独立 Worker 进程执行 OCR

## 依赖

- FuckClassroom: `>=0.1,<0.3`
- Plugin API: `1`
- Required plugin: `processing`
- Python dependency: `rapidocr-onnxruntime>=1.2`

AI OCR **不是硬依赖**。当 AI Summary 插件已启用且配置了 AI API Key 时，OCR Worker 通过 Plugin RPC `ai_summary.ocr.image` 调用视觉模型；否则 RapidOCR/Tesseract 仍可独立工作。

## 开发

开发分支为 `plugin-management`。合并到 `main` 后，CI 成功会自动发布 Registry v1 beta Release。
