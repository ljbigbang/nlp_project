import requests as re
import streamlit as st
import asyncio as asy
import ollama
import json

#url="http://localhost:11434/api/chat"
#ollama.pull('llama3.1')
async def chat():
    st.set_page_config(layout="wide", page_title="Chat with deep personalized character from The Big Bang")
    st.title("💬 Chat with Sheldon from The Big Bang！")
    st.caption("🦙 A deep personalized character powered by llama3")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "Sheldon", "content": "Hi, Guys!"}]
    for msg in st.session_state.messages:
        if msg["role"]=="Sheldon":
            st.chat_message(msg["role"],avatar="profile_sheldon.jpg").write(msg["content"])
        else:
            st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        message = {'role': 'user','content': 'Please act as Sheldon in The Big Bang and chat with me.'+prompt}
        with st.spinner("Thinking..."):
            response = await ollama.AsyncClient().chat(model='llama3.1',messages=[message],options={'temperature': 0.5},stream=False)
            st.session_state.messages.append({"role": "Sheldon", "content": response["message"]["content"]})
            st.chat_message("Sheldon",avatar="profile_sheldon.jpg").write(response["message"]["content"])
        # async for response in await ollama.AsyncClient().chat(model='llama3.1',messages=[message],options={'temperature': 0.5},stream=True):
        #     st.session_state.messages.append({"role": "Sheldon", "content": response["message"]["content"]})
        #     st.chat_message("Sheldon").write_stream(response["message"]["content"])
asy.run(chat())
# def llama3(prompt):
#     data = {
#         "model": "llama3",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ],
#         "stream": False,
#     }
#     headers = {
#         "Content-Type": "application/json"
#     }
#     response = re.post(url, headers=headers, json=data)
#     return response.json()["message"]["content"]

# st.set_page_config(layout="wide", page_title="Chat with deep personalized character from The Big Bang")
# st.title("💬 Chat with Sheldon from The Big Bang！")
# st.caption("🦙 A deep personalized character powered by llama3")
# if "messages" not in st.session_state:
#     st.session_state["messages"] = [{"role": "Sheldon", "content": "Hi, Guys!"}]
# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])
# if prompt := st.chat_input():
    #st.session_state.messages.append({"role": "user", "content": prompt})
    #st.chat_message("user").write(prompt)
    #response = llama3("Please act as Sheldon in The Big Bang and chat with me."+ prompt)
    #st.session_state.messages.append({"role": "Sheldon", "content": response})
    #st.chat_message("Sheldon").write(response)
#while(1):
    #user_prompt = input()
    #response = llama3("Please act as Sheldon in The Big Bang and chat with me."+ user_prompt)
    #print(response)



# 按间距中的绿色按钮以运行脚本。
# if __name__ == '__main__':


# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
