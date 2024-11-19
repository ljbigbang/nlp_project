import os
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
from streamlit_image_select import image_select
# API每天更新
os.environ["NVIDIA_API_KEY"] = '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'
#读取预处理的json数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
#读取GPT-4生成问题
def read_generated_questions(file_path):
    questions = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 假设每一行的格式为 "问题\t类别"
            question = line.strip().split('——')[0]
            category = line.strip().split('——')[1]
            questions.append((question,category))
    return "\n\n".join(f"question_category:{question_category}" for question_category in questions)
#利用ragchain,context得到query答案
def ragchain_result(chain,context,query):
    result = ''
    chunk_stream = chain.invoke({
        'question': query,
        'context':context,
    })
    for chunk in chunk_stream:
        result = result + chunk
    return result,chunk_stream

#建立llama3分类器
def question_classifier(question,context):
    question_classify_prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
                You are an intelligent classifier to classifying questions into one of the following four categories: 'personal background', 'research interest', 'publication' and 'recruitment'. You must learn the correspondence of the question and its category in the context and then classify.
                Context:{context}  
                Question: {question}
                Output format: Only output one phrase in the format of <class name>
                """
            ),
        ]
    )
    question_classify_llm = ChatNVIDIA(model='meta/llama-3.2-3b-instruct')
    question_classify_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | question_classify_prompt
            | question_classify_llm
            | StrOutputParser()
    )
    result, chunk_stream = ragchain_result(question_classify_chain,context,question)
    return result.strip("'").strip("<").strip(">")

def build_ragchain(model_link,question,question_context,data):
    category=question_classifier(question,question_context)
    print(f"类别:{category}")
    context=data[category]
    llm = ChatNVIDIA(model=model_link)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                'system',
                """
                You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the complete answer, combine your background knowledge to give a answer which is related to the Hong Kong Polytechnic University. Use five sentences maximum and keep the answer concise.
                Question: {question}
                Context: {context}
                """
            ),
        ]
    )
    rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain,context

data=load_json('./summary_summary_json/context_final.json')
generated_questions=read_generated_questions('./generated_question.txt')
#UI
st.set_page_config(layout="wide", page_title="chatrobot", theme="dark")
st.title("PolyRAS: A chatrobot for analyzing PolyU's researcher information 💬")
st.header("First, let us choose a large language model.")
with st.container():
    model_dic={
        #目前选用模型为llama3系列
        'llama-3.2-3b-instruct':'meta/llama-3.2-3b-instruct',
        'llama3-8b-instruct':'meta/llama3-8b-instruct',
        'llama-3.1-nemotron-51b-instruct':'nvidia/llama-3.1-nemotron-51b-instruct',
        'llama-3.1-70b-instruct':"meta/llama-3.1-70b-instruct",
        'llama-3.1-405b-instruct':"meta/llama-3.1-405b-instruct",
    }
    available_models=list(model_dic.keys())
    models_image_path=['model_images/'+ item + '.jpeg' for item in available_models]
    model_index = image_select(label="Models", images=models_image_path, captions=available_models,
                                use_container_width=False, return_value="index")
    model_label=available_models[model_index]
    model_link = model_dic[model_label]
    print(model_link)
st.header("Now, we can start to chat!")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "PolyRAS", "content": "Hi, Guys!"}]
for msg in st.session_state.messages:
    if msg["role"] == "PolyRAS":
        st.chat_message(msg["role"], avatar="polyu.jpg").write(msg["content"])
    else:
        st.chat_message(msg["role"]).write(msg["content"])
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    st.chat_message("user").write(input)
    with st.spinner("Thinking..."):
        rag_chain,context=build_ragchain(model_link, input, generated_questions, data)
        result,chunk_stream=ragchain_result(rag_chain,context,input)
        st.session_state.messages.append({"role": "PolyRAS", "content": result})
        print(st.session_state.messages)
        st.chat_message("PolyRAS", avatar="polyu.jpg").write(result)
