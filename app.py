import gradio as gr
import numpy as np
import time

def fake_diffusion(steps):
    for i in range(steps):
        time.sleep(1)
        image = np.random.random((600, 600, 3))
        yield image
    image = np.ones((1000,1000,3), np.uint8)
    image[:] = [255, 124, 0]
    yield image


demo = gr.Interface(fake_diffusion, inputs=gr.Slider(1, 10, 3), outputs="image")

demo.launch()