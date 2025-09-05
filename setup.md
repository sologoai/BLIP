### 启动服务

因模型文件太大，无法使用git管理，启动服务前需手动上传项目文件和模型文件（模型文件已存在于项目目录中 ./models/model_base_caption_capfilt_large.pth）

```
cd webui_blip/

docker build -t webui_blip .

docker run --name blip_web_api -p 52001:52001 -d webui_blip
```

### 服务停止/启动

```

docker stop blip_web_api

docker start blip_web_api

```

### 测试demo

```sh

curl -X POST -H "Content-Type: application/json" -d '{"img_urls": ["https://pro.upload.logomaker.com.cn/24/01/06/5ce553a276204a531e79b802469e263d.jpeg"]}' 'http://localhost:52001/post'


```