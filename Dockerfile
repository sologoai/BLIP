# 使用 PyTorch 镜像作为基础镜像
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 将工作目录设为 /workspace
WORKDIR /workspace

# 将本地的 文件内容复制到容器的工作目录
COPY . .

RUN pip install -r requirements.txt

# 运行 Flask 服务
CMD ["nohup", "python", "service.py","&"]
