import time

import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from llama_index.postprocessor.cohere_rerank import CohereRerank

from chatbot import load_index, answer_question


def start_session(): # user-specific session
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    
    if "chat_limit" not in st.session_state:
        st.session_state["chat_limit"] = 0


@st.cache_resource(show_spinner=False) # shared between different users
def load_resources():
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    index = load_index()
    rerank = CohereRerank(api_key=st.secrets["OPENAI_API_KEY"], top_n=3, max_retries=3)
    
    return client, index, rerank


def stream_answer(answer):
    for word in answer.split(" "):
        yield word + " "
        time.sleep(0.04)


def run():
    #st.title("Alexander Weyhe - CV Chatbot")
    st.markdown(
    """
    <div class="header">
        <div>
            <h1>Alexander Weyhe</h1>
            <p>NLP & LLM Enthusiast</p>
        </div>
        <div class="links">
            <a href="https://github.com/AlexWeyhe" target="_blank">GitHub</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

        
    start_session()

    
    with st.spinner("Loading chatbot..."):
        client, index, rerank = load_resources()
    
    for message in st.session_state["chat_history"]:
        avatar_path = "extras/avatar.png" if message["role"] == "assistant" else None
        
        with st.chat_message(message["role"], avatar=avatar_path):
            st.markdown(message["content"])
    
    prompt = st.chat_input("Stelle eine Frage zu Alex' Lebenslauf.\n"
                            "Ask a question about Alex' CV.",
                            disabled=st.session_state["chat_limit"] >= 10)
        
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = answer_question(prompt=prompt,
                                client=client,
                                chat_history=st.session_state["chat_history"],
                                rerank=rerank,
                                index=index)
            
        with st.chat_message("assistant", avatar="extras/avatar.png"):
            st.write_stream(stream_answer(answer))
            
        st.session_state["chat_limit"] += 1
    
    if st.session_state["chat_limit"] >= 10:
        st.info("Du hast das Limit von 10 Fragen erreicht. Schreib mir gerne, wenn du mehr wissen möchstest.\n"
                "You reached the limit of 10 questions. Feel free to reach out to me, for further questions.")


if __name__ == "__main__":
    run()
