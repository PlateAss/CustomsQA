import gradio as gr

def combine(a, b):
    return a + " " + b
with gr.Blocks() as demo:
    txt = gr.Textbox(label="Input", lines=2)
    txt_2 = gr.Textbox(label="Input 2")
    txt_3 = gr.Textbox(value="", label="Output")
    btn = gr.Button(value="Submit")
    btn.click(combine, inputs=[txt, txt_2], outputs=[txt_3])
demo.launch()