#FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
# 设置工作目录
WORKDIR /app

# 更新源并安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install paddlepaddle-gpu==2.6.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

#RUN cp /usr/local/cuda-11.7/targets/x86_64-linux/lib/libcublas.so.11 /usr/lib/libcublas.so
RUN cp /usr/local/cuda-12.0/targets/x86_64-linux/lib/libcublas.so.12 /usr/lib/libcublas.so
RUN cp /usr/lib/x86_64-linux-gnu/libcudnn.so.8  /usr/lib/libcudnn.so
# 复制项目文件到容器中
COPY . .

# 暴露必要的端口 (根据您的应用程序设置, 假设为7860)
EXPOSE 7860

# 设置容器启动时运行的命令
CMD ["python3", "main.py"]