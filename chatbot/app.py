from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
#langsmith tracking
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')

#define prompt template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

##streamlit framework
st.title('Langchain Demo With OPENAI API')
input_text=st.text("search the topic you want ")

#open AI LLM
llm=ChatOpenAI(model='gpt-3.5-turbo')
outputparser=StrOutputParser()  #responsible for getting output
chain=prompt|llm|outputparser

if input_text:
    st.write(chain.invoke({"question":input_text}))