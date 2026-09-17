# OCR fallback ingestion v1

## 目标

OCR fallback 为扫描 PDF 和图片补充文本提取能力，使恢复出的文本继续进入现有的
`DocumentLoader → Chunk → Embedding → Milvus` 链路。它不改变分块、向量、Milvus schema、
检索、回答或会话记忆。

## 执行流程

PDF 始终先使用原有 PDFium/PyPDF loader。仅当开启 `OCR_ENABLED` 且某页有效字符数低于
`OCR_MIN_TEXT_THRESHOLD` 时，该页才渲染为图片并交给 PaddleOCR。正常文本页保持原文本与
原 metadata；识别成功的低文本页使用 OCR 文本替换，并标记：

- `parser_backend=paddleocr`
- `page`：原 PDF loader 的 0-based 页码
- `location=page:<page>`
- `ocr_fallback=true`

图片文件（PNG、JPG/JPEG、BMP、TIFF、WebP）直接通过 PaddleOCR，输出 0-based 逻辑页和
`location=image:0`。OCR 返回统一 `ParsedDocument / ParsedUnit`，再转换为 LangChain
`Document`，因此后续沿用现有 chunk 和入库实现。

## 配置

```env
OCR_ENABLED=false
OCR_LANG=auto
OCR_MIN_TEXT_THRESHOLD=100
```

`auto` 使用可同时识别中文、英文和数字的中文模型。默认关闭意味着升级后已有 PDF/TXT
行为不变。首次启用前安装可选依赖，并按机器环境安装兼容的 PaddlePaddle 推理引擎：

```powershell
pip install -e ".[ocr]"
```

该可选组默认提供 CPU 推理引擎。GPU 版本需按 PaddlePaddle 官方安装说明选择与 CUDA 匹配的
wheel。首次实际 OCR 可能下载
模型，生产环境应提前完成模型预热。

## 回退与风险

- OCR 只补救低文本页，不处理已有充足文字的 PDF 页。
- OCR 失败或未识别到文字时保留旧 loader 的原页面结果，不伪造文本。
- 有文字但字数很少的封面也可能触发 OCR，可通过阈值调整。
- OCR 只恢复文本和页码，不恢复表格单元格结构；表格结构化仍属于现有 TableStore 范围。
- 扫描页渲染为 2× 图片，长 PDF 的首次导入时间和内存使用会增加，但只处理被判为低文本的页。

## 覆盖范围

该版本扩展了中文扫描图片、英文扫描图片及图片型 PDF 页的非结构化文本覆盖范围。
图片与 OCR PDF 文本可进入现有 chunk pipeline；没有修改检索和回答行为。
