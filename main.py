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
        self.ocr = None  # 初始不加载模型

    def load_model(self, det_model_dir, rec_model_dir, cls_model_dir):
        if self.ocr is None:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                show_log=False,
                use_gpu=True,
                det_model_dir=det_model_dir,
                rec_model_dir=rec_model_dir,
                cls_model_dir=cls_model_dir,
            )
            print("模型已加载")

    def unload_model(self):
        if self.ocr is not None:
            del self.ocr
            self.ocr = None
            gc.collect()
            print("模型已释放")

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
                if not self.ocr_processor.ocr:  # 检查是否加载了模型
                    time.sleep(1)
                    continue
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

    @staticmethod
    def get_subdirectories(base_dir):
        dir_path = os.path.join(base_dir)
        return [name for name in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, name))]


if __name__ == "__main__":
    processor = Processor()
    worker_thread = threading.Thread(target=processor.background_worker)
    worker_thread.daemon = True  # 设置为守护线程, 主程序退出时线程自动退出
    worker_thread.start()
    with gr.Blocks() as demo:
        gr.Markdown("# PDF OCR")
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                upload_files = gr.File(
                    file_count="multiple",
                    file_types=[".pdf"],
                    label="需要处理的文件拖入此",
                )
                # 模型选择下拉列表
                det_model_choices = GradioFunctions.get_subdirectories('models/det_model_dir')
                rec_model_choices = GradioFunctions.get_subdirectories('models/rec_model_dir')
                cls_model_choices = GradioFunctions.get_subdirectories('models/cls_model_dir')

                det_model_dir_dropdown = gr.Dropdown(
                    label="选择检测模型目录",
                    choices=det_model_choices,
                    value='ch_PP-OCRv4_det_infer' if 'ch_PP-OCRv4_det_infer' in det_model_choices else
                    det_model_choices[0]
                )

                rec_model_dir_dropdown = gr.Dropdown(
                    label="选择识别模型目录",
                    choices=rec_model_choices,
                    value='ch_PP-OCRv4_rec_infer' if 'ch_PP-OCRv4_rec_infer' in rec_model_choices else
                    rec_model_choices[0]
                )

                cls_model_dir_dropdown = gr.Dropdown(
                    label="选择分类模型目录",
                    choices=cls_model_choices,
                    value='ch_ppocr_mobile_v2.0_cls_infer' if 'ch_ppocr_mobile_v2.0_cls_infer' in cls_model_choices else
                    cls_model_choices[0]
                )

                with gr.Row(equal_height=True):
                    load_model_button = gr.Button("加载模型")
                    unload_model_button = gr.Button("释放模型")
                    process_button = gr.Button(
                        value="开始处理",
                        variant="primary",
                    )

                # 模型状态显示
                model_status = gr.Textbox(
                    label="模型状态",
                    value="模型未加载",
                    interactive=False,
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
                value=GradioFunctions.get_file_from_result(),
            )


        def load_model_function(det_model_selection, rec_model_selection, cls_model_selection):
            det_model_dir = os.path.join('models', 'det_model_dir', det_model_selection)
            rec_model_dir = os.path.join('models', 'rec_model_dir', rec_model_selection)
            cls_model_dir = os.path.join('models', 'cls_model_dir', cls_model_selection)
            processor.ocr_processor.load_model(det_model_dir, rec_model_dir, cls_model_dir)
            return "模型已加载"


        def unload_model_function():
            processor.ocr_processor.unload_model()
            return "模型未加载"


        def timer_update_func():
            return (
                GradioFunctions.get_file_from_result(),  # 刷新处理结果
                "\n".join(GradioFunctions.get_files_from_queue()),  # 刷新待处理队列
                processing_filename + " " + str(current_progress) + "%",  # 刷新正在处理
            )


        process_button.click(GradioFunctions.save_file, inputs=upload_files)
        load_model_button.click(
            fn=load_model_function,
            inputs=[det_model_dir_dropdown, rec_model_dir_dropdown, cls_model_dir_dropdown],
            outputs=model_status
        )
        unload_model_button.click(
            fn=unload_model_function,
            inputs=None,
            outputs=model_status
        )
        timer = gr.Timer(1)
        timer.tick(fn=timer_update_func, inputs=None, outputs=[finished_files, processing_queue, processing_file])
    demo.launch(server_name="0.0.0.0", share=False)
