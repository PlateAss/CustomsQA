import os
xLab=0
modelName="internlm2-chat-7b"
modelpath=[f"/root/share/model_repos/{modelName}",f"/home/xlab-app-center/{modelName}"]
sentencePath=["/root/data/model/sentence-transformer","/home/xlab-app-center/sentence-transformer"]
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
if os.path.isdir(f"{sentencePath[xLab]}")==False:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.system(f'huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir {sentencePath[xLab]}')

from lmdeploy import turbomind as tm

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

from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings


# 定义 Embeddings
embeddings = HuggingFaceEmbeddings(model_name=sentencePath[xLab])
# 加载数据库
vectordb = Chroma(persist_directory='./chroma', embedding_function=embeddings)

from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
class InternLM_LLM(LLM):

    def __init__(self):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()

    def _call(self, prompt : str, stop: Optional[List[str]] = None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any):
        # 重写调用函数
        system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
        - InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
        - InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
        """
        
        # messages = [(system_prompt, '')]
        # response, history = generator.chat(self.tokenizer, prompt , history=messages)

        prompt_t = tm_model.model.get_prompt(prompt)
        input_ids = tm_model.tokenizer.encode(prompt_t)
        for outputs in generator.stream_infer(session_id=0,input_ids=[input_ids]):            
            response = tm_model.tokenizer.decode(outputs[1])            
        return response
        
    @property
    def _llm_type(self) -> str:
        return "InternLM"
    
llm = InternLM_LLM()

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
# 我们所构造的 Prompt 模板
template = """使用以下上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
问题: {question}
可参考的上下文：
···
{context}
···
如果给定的上下文无法让你做出回答，请回答你不知道。
有用的回答:"""

# 调用 LangChain 的方法来实例化一个 Template 对象，该对象包含了 context 和 question 两个变量，在实际调用时，这两个变量会被检索到的文档片段和用户提问填充
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)



qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

import gradio as gr
with gr.Blocks(title="海关问答") as demo:    
    chatbot = gr.Chatbot(show_label=False)
    with gr.Row():
        with gr.Column(scale=5):
            msg = gr.Textbox(label="输入你的问题")
        with gr.Column(scale=1,min_width=80):        
            submit = gr.Button("发送")
            clear = gr.Button("清除")
        with gr.Column(scale=1,min_width=80):
            chkdb = gr.Checkbox(True,label="知识库")    

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history,chkdb):
        print(chkdb)
        if chkdb==True:
            history[-1][1]=""
            result = qa_chain({"query": history[-1][0]})
            history[-1][1]=result["result"]
            yield history
        else:
            history[-1][1] = ""
            prompt = tm_model.model.get_prompt(combine_history(history))
            input_ids = tm_model.tokenizer.encode(prompt)
            for outputs in generator.stream_infer(session_id=0,input_ids=[input_ids]):            
                response = tm_model.tokenizer.decode(outputs[1])
                history[-1][1] = response
                yield history

    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot,chkdb], chatbot)
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot,chkdb], chatbot)
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
