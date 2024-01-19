import gradio as gr
import os

# 输入一张图片，旋转45°后输出
def image_mod(image):
    return image.rotate(45)


demo = gr.Interface(image_mod, gr.Image(type="pil"), "image",
    allow_flagging="never")

if __name__ == "__main__":
    demo.launch()