# CustomsQA 外贸小助手
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://beta.openxlab.org.cn/apps/detail/mmpose/RTMPose)
![](assets/t1.png)

## 简介

使用上海人工智能实验室的Internlm2-chat-7b模型为基础，实现外贸相关知识的问答。

## 介绍

**知识库包含的内容：**

- 中华人民共和国海关法
- 中华人民共和国海关关衔条例
- 中华人民共和国船舶吨税法
- 中华人民共和国国境卫生检疫法
- 中华人民共和国进出口商品检验法
- 中华人民共和国进出境动植物检疫法
- 中华人民共和国食品安全法
- 国家代码
- 货币代码
- 运输方式代码
- 监管方式代码
- 关区代码
- 贸易便利化术语
- HS商品编码

**代码说明：**
app.py为代码主体
requirements.txt为需要安装的python库
chroma文件夹是根据上述内容生成的向量库

**运行方式：**

```
pip install -r requirements.txt
python app.py
```


