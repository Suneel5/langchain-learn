# Q&A Chatbot
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
#from dotenv import load_dotenv
#load_dotenv()  # take environment variables from .env.
import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain_community.llms import Ollama

## Function to load OpenAI model and get respones
# Load environment variables from .env file
load_dotenv()

#ascess enviroment variables
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')

##initialize our streamlit app

st.set_page_config(page_title="Q&A Demo")
st.header("celibrity search")

input=st.text_input("search the topic you want to:  ")

first_input_prompt=PromptTemplate(
    input_variables=['name'],
    template='tell me about {name}'
)
person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory=ConversationBufferMemory(input_key='person',memory_key='chat_history')
events_memory=ConversationBufferMemory(input_key='dob',memory_key='events_history')

llm=Ollama(model='llama2:latest')
chain1=LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

second_input_prompt=PromptTemplate(
    input_variables=['person'],
    template='when was  {person} born?'
)
chain2=LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)

third_input_prompt=PromptTemplate(
    input_variables=['dob'],
    template='what was the 5 major events happen around {dob} in the world'
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='events',memory=events_memory)
parent_chain=SequentialChain(chains=[chain1,chain2,chain3],input_variables=['name'],output_variables=['person','dob','events'],verbose=True)

if input:
    st.write(parent_chain({'name':input}))
    with st.expander('Person name'):
        st.info(person_memory.buffer)
    with st.expander('major evnets'):
        st.info(events_memory.buffer)




