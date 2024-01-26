import os
xlab=0
if os.path.isdir("/home/xlab-app-center"):
    xlab=1
from openxlab.model import download
if xlab==1:
    download(model_repo='OpenLMLab/internlm2-chat-7b', output='/home/xlab-app-center/internlm2-chat-7b')
os.system("lmdeploy serve gradio /home/xlab-app-center/internlm2-chat-7b --model-name internlm2-chat-7b --server-port 7860")
# from dataclasses import asdict
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tools.transformers.interface import GenerationConfig, generate_interactive
# import gradio as gr

# def load_model():
#     model_name_or_path = "/root/share/model_repos/internlm-chat-7b"
#     if xlab==1:
#         model_name_or_path = "/home/xlab-app-center/internlm-chat-7b"
#     model = (AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
#     .to(torch.bfloat16)
#     .cuda())
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
#     return model, tokenizer

# def prepare_generation_config():
#     generation_config = GenerationConfig(max_length=2048, top_p=0.8, temperature=0.7)
#     return generation_config

# user_prompt = "<|User|>:{user}\n"
# robot_prompt = "<|Bot|>:{robot}<eoa>\n"
# cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"


# def combine_history(history):
#     total_prompt = ""
#     for msg in history:
#         cur_prompt = user_prompt.replace("{user}", msg[0])
#         total_prompt += cur_prompt
#         cur_prompt = robot_prompt.replace("{robot}", msg[1])
#         total_prompt += cur_prompt
#     total_prompt += cur_query_prompt.replace("{user}", history[-1][0])
#     return total_prompt

# model, tokenizer = load_model()
# generation_config = prepare_generation_config()
# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.Button("清除")

#     def user(user_message, history):
#         return "", history + [[user_message, None]]

#     def bot(history):
#         history[-1][1] = ""
#         for character in generate_interactive(model=model,tokenizer=tokenizer,prompt=combine_history(history),additional_eos_token_id=103028,**asdict(generation_config)):
#             history[-1][1] = character
#             yield history

#     msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
#     clear.click(lambda: None, None, chatbot, queue=False)
# demo.queue()
# demo.launch()
