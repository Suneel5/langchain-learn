from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#ascess enviroment variables
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
st.title('Langchain Demo With LLAMLA2 API')
input_text=st.text_input("search the topic you want: ")

#open AI LLM
llm=Ollama(model='llama2:latest')
outputparser=StrOutputParser()  #responsible for getting output(parses llm result)
chain=prompt|llm|outputparser

if input_text:
    st.write(chain.invoke({"question":input_text}))