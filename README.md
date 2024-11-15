# PDF OCR 文本提取工具

本项目是一个基于 PaddleOCR 和 Gradio 的 PDF 文本提取工具, 旨在将 PDF 文件中的文字内容识别并提取为可编辑的文本文件。

## 功能特点

- **批量处理** : 支持同时上传多个 PDF 文件进行 OCR 识别。
- **实时进度显示** : 在处理过程中, 实时显示当前正在处理的文件名和处理进度, 以及待处理的文件队列。
- **结果导出** : 处理完成后, 识别的文本结果将以 `.txt` 文件形式保存, 可直接下载。
- **高精度识别** : 利用 PaddleOCR 强大的 OCR 识别能力, 支持中英文字符的准确识别。

## 使用方法

1. **环境要求** : 
   - Python 3.x
   - PaddleOCR
   - Gradio
   - 详见 `requirements.txt`

2. **安装依赖** : 
   ```bash
   pip install -r requirements.txt
   ```

3. **准备模型文件** : 
   - 将所需的 PaddleOCR 模型文件放置到指定目录: 
     - 检测模型: `/app/ch_PP-OCRv4_det_infer`
     - 识别模型: `/app/ch_PP-OCRv4_rec_infer`
     - 方向分类模型: `/app/ch_ppocr_mobile_v2.0_cls_infer`

4. **运行程序** : 
   ```bash
   python main.py
   ```
   或使用 Docker: 
   ```bash
   docker-compose up
   ```

5. **访问界面** : 
   - 在浏览器中打开 `http://localhost:7860`, 进入 Gradio 用户界面。

6. **操作步骤** : 
   - **上传文件** : 将需要处理的 PDF 文件拖拽到界面的文件上传区域。
   - **开始处理** : 点击“开始处理”按钮, 程序将开始对上传的 PDF 文件进行 OCR 识别。
   - **查看进度** : 实时查看当前处理的文件名 、 进度百分比和待处理的文件队列。
   - **下载结果** : 处理完成后, 在“处理结果”区域下载生成的文本文件。

## 项目结构

- `main.py`: 主程序代码, 包含文件上传处理 、 OCR 识别和结果导出等功能。
- `requirements.txt`: 项目依赖的 Python 库列表。
- `Dockerfile`: 用于构建 Docker 镜像的配置文件。
- `docker-compose.yml`: Docker Compose 配置文件, 方便快速启动项目。
- `models`: 存放 OCR 模型文件的目录 (需自行添加) 。
- `uploads`: 默认的文件上传目录 (程序自动创建) 。
- `results`: OCR 识别结果保存目录 (程序自动创建) 。
