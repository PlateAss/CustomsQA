import os
from openxlab.model import download
download(model_repo='OpenLMLab/internlm2-chat-7b', output='/home/xlab-app-center/internlm-chat-7b')
os.system("lmdeploy serve gradio /home/xlab-app-center/internlm-chat-7b --model-name internlm-chat-7b --server-port 7860")
# import gradio as gr
# import random
# import time

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.Button("清除")

#     def respond(message, chat_history):
#         bot_message = random.choice(["你好吗？", "我爱你", "我很饿"])
#         chat_history.append((message, bot_message))
#         time.sleep(1)
#         return "", chat_history

#     msg.submit(respond, [msg, chatbot], [msg, chatbot])
#     clear.click(lambda: None, None, chatbot, queue=False)

# demo.launch()
