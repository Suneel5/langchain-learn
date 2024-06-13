import streamlit as st
import os
from langchain_groq import  ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain 
from langchain_community.vectorstores import FAISS 
from dotenv import load_dotenv
load_dotenv()

#load groq api key

groq_api_key=os.getenv('GROQ_API_KEY')
if "vector" not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loaders=WebBaseLoader('https://python.langchain.com/v0.1/docs/modules/tools/')
    st.session_state.docs=st.session_state.loaders.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("Chatgroq demo")
llm=ChatGroq(api_key=groq_api_key,model_name="Gemma-7b-It")

prompt=ChatPromptTemplate.from_template(
    """Answer to questions based on provided context only.
        please provide most accutate based on the question
        <context>
        {context}
        </context>
        questions:{input}
    """

)

document_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.vectors.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)
prompt=st.text_input("Enter your prompt here: ")
if prompt:
    response=retrieval_chain.invoke({'input': prompt})
    st.write(response['answer'])







  