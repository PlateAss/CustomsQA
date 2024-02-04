import os
xLab=0
modelName="internlm2-chat-7b"
modelPath=[f"/root/share/model_repos/{modelName}",f"/home/xlab-app-center/{modelName}"]
sentencePath=["/root/data/model/sentence-transformer","/home/xlab-app-center/sentence-transformer"]
if os.path.isdir("/home/xlab-app-center"):
    xLab=1
from openxlab.model import download
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
if xLab==1:
    if os.path.isdir(f"{modelPath[xLab]}")==False:
        download(model_repo=f'OpenLMLab/{modelName}', output=f"{modelPath[xLab]}")        
        os.system(f"lmdeploy lite auto_awq {modelPath[xLab]} --work-dir {modelPath[xLab]}-4bits")
    # os.system(f"lmdeploy serve gradio {modelpath[xLab]}-4bits --server-port 7860 --model-format awq --backend turbomind")
else:
    if os.path.isdir(f"./{modelName}-4bits")==False:        
        os.system(f"lmdeploy lite auto_awq {modelPath[xLab]} --work-dir {modelName}-4bits")
    # os.system(f"lmdeploy serve gradio ./{modelName}-4bits --server-port 7860 --model-format awq --backend turbomind")
if os.path.isdir(f"{sentencePath[xLab]}")==False:    
    os.system(f'huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir {sentencePath[xLab]}')
os.system(f'lmdeploy serve api_server internlm2-chat-7b-4bits --model-name internlm2-chat-7b')
# from lmdeploy import turbomind as tm

# tm_model = tm.TurboMind.from_pretrained(f"{modelName}-4bits", model_name=modelName,trust_remote_code=True)
# generator = tm_model.create_instance()

__import__('pysqlite3')
import sys
sys.modules['sqlite3']=sys.modules.pop('pysqlite3')
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# 定义 Embeddings
embeddings = HuggingFaceEmbeddings(model_name=sentencePath[xLab])
# 加载数据库
vectordb = Chroma(persist_directory='./chroma', embedding_function=embeddings)

# from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
# class InternLM_LLM(LLM):

#     def __init__(self):
#         # model_path: InternLM 模型路径
#         # 从本地初始化模型
#         super().__init__()

#     def _call(self, prompt : str, stop: Optional[List[str]] = None,
#                 run_manager: Optional[CallbackManagerForLLMRun] = None,
#                 **kwargs: Any):
#         # 重写调用函数
#         prompt_t = tm_model.model.get_prompt(prompt)        
#         input_ids = tm_model.tokenizer.encode(prompt_t)
#         for outputs in generator.stream_infer(session_id=0,input_ids=[input_ids]):            
#             response = tm_model.tokenizer.decode(outputs[1])            
#         return response
        
#     @property
#     def _llm_type(self) -> str:
#         return "InternLM"
    
# llm = InternLM_LLM()
from langchain_openai import ChatOpenAI
os.environ['OPENAI_API_KEY'] = 'none'
os.environ['OPENAI_BASE_URL'] = 'http://0.0.0.0:23333/v1'
llm = ChatOpenAI(model_name="internlm2-chat-7b")
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import (AIMessage)
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

import gradio as gr
with gr.Blocks(title="外贸小助手") as demo:    
    chatbot = gr.Chatbot(show_label=False)
    with gr.Row():
        with gr.Column(scale=5):
            msg = gr.Textbox(label="输入你的问题")
        with gr.Column(scale=1,min_width=80):        
            submit = gr.Button("发送")
            clear = gr.Button("清除")
        with gr.Column(scale=1,min_width=80):
            chkdb = gr.Checkbox(True,label="知识库")
            if xLab==1:
                debug = gr.Checkbox(False,label="调试模式",visible=False)
            else:
                debug = gr.Checkbox(True,label="调试模式")   

    def user(user_message, history):
        if len(history)>10:
            del history[0]
        return "", history + [[user_message, None]]

    def bot(history,chkdb,debug):
        if debug==True:
            history[-1][1]=""
            result = qa_chain({"query": history[-1][0]})
            response=llm.invoke([combine_history(history)])
            history[-1][1]=f"**检索知识库回答**:{result['result']}\n\n**大模型回答**:{response.content}"
            yield history
        else:
            if chkdb==True:
                history[-1][1]=""
                result = qa_chain({"query": history[-1][0]})
                history[-1][1]=result["result"]
                yield history
            else:
                history[-1][1] = ""
                response=llm.invoke([combine_history(history)])
                history[-1][1]=response.content
                yield history
    submit.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot,chkdb,debug], chatbot)
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, [chatbot,chkdb,debug], chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)
demo.queue()
demo.launch()
