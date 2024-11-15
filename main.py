# -*- coding: utf-8 -*-
import threading
import time
import gradio as gr
import os
import shutil
import queue
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from tqdm import tqdm
import fitz
import gc

current_progress = 0  # 当前进度百分比
file_queue = queue.Queue()
processing_filename = ""  # 当前正在处理的文件名

os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)


class OCRProcessor:
    def __init__(self):
        # 初始化 OCR 模型, 确保使用 GPU
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
            use_gpu=True,
            det_model_dir="/app/ch_PP-OCRv4_det_infer",
            rec_model_dir="/app/ch_PP-OCRv4_rec_infer",
            cls_model_dir="/app/ch_ppocr_mobile_v2.0_cls_infer",
        )

    @staticmethod
    def get_pdf_page_count(pdf_path):
        """快速获取PDF文件的页数"""
        doc = fitz.open(pdf_path)  # 打开PDF文件
        page_count = doc.page_count  # 获取页数
        doc.close()  # 关闭文档
        return page_count

    @staticmethod
    def pdf_to_image(pdf_path, page_number):
        """将PDF文件的指定页转换为图像"""
        images = convert_from_path(
            pdf_path, first_page=page_number, last_page=page_number
        )
        return images[0] if images else None

    def ocr_image(self, image):
        """对单个图像应用OCR, 返回识别的文本"""
        text = ""
        image = image.convert("RGB")
        image = np.array(image)
        result = self.ocr.ocr(image, cls=True)
        try:
            if result != [None]:
                for line in result:
                    for info in line:
                        text += info[1][0] + "\n"
                text += "\n\n"
            else:
                print("识别到空页")
        except Exception as e:
            print(str(e))
        # 释放资源
        del image
        del result
        gc.collect()
        return text

    def ocr_processor(self, file_name):
        """处理单个文件的主流程"""
        global current_progress
        pdf_path = os.path.join("uploads", file_name)
        current_progress = 0
        total_pages = self.get_pdf_page_count(pdf_path)
        print(f"Processing: {pdf_path} 总页数: {total_pages}")
        book_name = os.path.splitext(os.path.basename(pdf_path))[0]
        txt_path = os.path.join("results", f"{book_name}.txt")
        page_texts = []
        for page_num in tqdm(range(1, total_pages + 1)):
            image = self.pdf_to_image(pdf_path, page_num)
            if image:
                page_text = self.ocr_image(image)
                page_texts.append(page_text)
                del image
                gc.collect()
            else:
                print(f"无法获取第 {page_num} 页的图像")
            current_progress = int(page_num * 100 // total_pages)

        with open(txt_path, "w", encoding="utf-8") as output_file:
            output_file.write("\n".join(page_texts))
        print(f"文本保存到: {txt_path}")


class Processor:
    def __init__(self):
        self.ocr_processor = OCRProcessor()

    def background_worker(self):
        global processing_filename
        while True:
            if not file_queue.empty():
                try:
                    processing_filename = file_queue.get()
                    self.ocr_processor.ocr_processor(processing_filename)
                except Exception as e:
                    print(f"处理文件 {processing_filename} 时出错: {e}")
                finally:
                    file_queue.task_done()
                    processing_filename = ""
            else:
                time.sleep(1)


class GradioFunctions:
    # 保存上传的文件, 同名文件覆盖
    @staticmethod
    def save_file(files):
        for file in files:
            shutil.copy(file, "uploads")
            file_name = os.path.basename(file)
            file_queue.put(file_name)

    @staticmethod
    def get_file_from_result() -> list:
        return sorted(
            [os.path.join("results", file_name) for file_name in os.listdir("results")],
            key=os.path.getmtime,
            reverse=True
        )

    @staticmethod
    def get_files_from_queue() -> list:
        return list(file_queue.queue)


with gr.Blocks() as demo:
    gr.Markdown("# PDF OCR")
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            upload_files = gr.File(
                file_count="multiple",
                file_types=[".pdf"],
                label="需要处理的文件拖入此",
            )
            process_button = gr.Button(
                value="开始处理",
            )
        with gr.Column(scale=1):
            # 当前正在处理(1个)
            processing_file = gr.Textbox(
                label="正在处理",
                interactive=False,
                value="",
            )
            processing_queue = gr.Textbox(
                label="待处理队列",
                interactive=False,
                value=lambda: "\n".join(GradioFunctions.get_files_from_queue()),
                max_lines=6,
            )
    with gr.Column(scale=1):
        finished_files = gr.File(
            label="处理结果",
            interactive=False,
            value=GradioFunctions.get_file_from_result,
        )

    def refresh_get_file_from_result():
        return GradioFunctions.get_file_from_result()

    def refresh_get_files_from_queue():
        return "\n".join(GradioFunctions.get_files_from_queue())

    def refresh_processing_file():
        return processing_filename + " " + str(current_progress) + "%"

    process_button.click(GradioFunctions.save_file, inputs=upload_files)
    timer = gr.Timer(1)
    timer.tick(fn=refresh_processing_file, inputs=None, outputs=processing_file)
    timer.tick(fn=refresh_get_files_from_queue, inputs=None, outputs=processing_queue)
    timer.tick(fn=refresh_get_file_from_result, inputs=None, outputs=finished_files)

if __name__ == "__main__":
    processor = Processor()
    worker_thread = threading.Thread(target=processor.background_worker)
    worker_thread.daemon = True  # 设置为守护线程, 主程序退出时线程自动退出
    worker_thread.start()
    demo.launch(server_name="0.0.0.0", share=False)
