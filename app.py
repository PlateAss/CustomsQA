import os
xLab=0
modelName="internlm2-chat-7b"
modelpath=[f"/root/share/model_repos/{modelName}",f"/home/xlab-app-center/{modelName}"]
if os.path.isdir("/home/xlab-app-center"):
    xLab=1
from openxlab.model import download
if xLab==1:
    if os.path.isdir(f"{modelpath[xLab]}")==False:
        download(model_repo=f'OpenLMLab/{modelName}', output=f"{modelpath[xLab]}")        
        os.system(f"lmdeploy lite auto_awq {modelpath[xLab]} --work-dir {modelpath[xLab]}-4bits")
    # os.system(f"lmdeploy serve gradio {modelpath[xLab]}-4bits --server-port 7860 --model-format awq --backend turbomind")
else:
    if os.path.isdir(f"./{modelName}-4bits")==False:        
        os.system(f"lmdeploy lite auto_awq {modelpath[xLab]} --work-dir ./{modelName}-4bits")
    # os.system(f"lmdeploy serve gradio ./{modelName}-4bits --server-port 7860 --model-format awq --backend turbomind")

from lmdeploy import turbomind as tm
import gradio as gr
user_prompt = "<|User|>:{user}\n"
robot_prompt = "<|Bot|>:{robot}<eoa>\n"
cur_query_prompt = "<|User|>:{user}<eoh>\n<|Bot|>:"

def combine_history(history):
    total_prompt = ""
    for msg in history:
        cur_prompt = user_prompt.replace("{user}", msg[0])
        total_prompt += cur_prompt
        cur_prompt = robot_prompt.replace("{robot}", msg[1])
        total_prompt += cur_prompt
    total_prompt += cur_query_prompt.replace("{user}", history[-1][0])
    return total_prompt

tm_model = tm.TurboMind.from_pretrained(f"{modelName}-4bits", model_name=modelName,trust_remote_code=True)
generator = tm_model.create_instance()
with gr.Blocks() as demo:
    gr.Label("海关问答",show_label=False)
    chatbot = gr.Chatbot(show_label=False)
    with gr.Row():
        with gr.Column(scale=5):
            msg = gr.Textbox(label="输入你的问题")
        with gr.Column(scale=1,min_width=80):        
            submit = gr.Button("发送")
            clear = gr.Button("清除")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        history[-1][1] = ""
        prompt = tm_model.model.get_prompt(combine_history(history))
        input_ids = tm_model.tokenizer.encode(prompt)
        for outputs in generator.stream_infer(session_id=0,input_ids=[input_ids]):            
            response = tm_model.tokenizer.decode(outputs[1])
            history[-1][1] = response
            yield history

    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)
demo.queue()
demo.launch()




    
# from dataclasses import asdict
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tools.transformers.interface import GenerationConfig, generate_interactive
# import gradio as gr

# def load_model():
#     model_name_or_path = modelpath[xlab]
#     model = (AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
#     .to(torch.float16)
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
