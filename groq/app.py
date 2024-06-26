import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# Load groq API key
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("GROQ_API_KEY not found in environment variables.")
    st.stop()

if "vectors" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings()
    st.session_state.loaders = WebBaseLoader('https://python.langchain.com/v0.1/docs/modules/tools/')
    st.session_state.docs = st.session_state.loaders.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    
    # Debug statement to verify documents are split
    if not st.session_state.final_documents:
        st.error("No documents were split. Check the document loading and splitting process.")
        st.stop()

    try:
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    except ValueError as e:
        st.error(f"Error while creating FAISS vectors: ou{e} he")
        st.stop()

st.title("Chatgroq demo")
llm = ChatGroq(api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt_template = """Answer to questions based on provided context only.
Please provide the most accurate answer based on the question.
<context>
{context}
</context>
Question: {input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_prompt = st.text_input("Enter your prompt here: ")
if user_prompt:
    try:
        response = retrieval_chain.invoke({'input': user_prompt})
        st.write(response['answer'])
    except Exception as e:
        st.error(f"Error during retrieval chain invocation: {e} ouu")
