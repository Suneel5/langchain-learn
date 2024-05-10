from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
#langsmith tracking
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')

# Initialize FastAPI app
app=FastAPI(
    title='Langhchain server',
    version='1.0',
    description='A server for the Langchain'
)


# Add routes to the app
add_routes(
    app,
    ChatOpenAI(),
    path='/openai'
)

#ollama models 
llm2=Ollama(model='llama2:latest')
llm3=Ollama(model='llama3:latest')

#prompt template for llama2
prompt1=ChatPromptTemplate.from_template("write essay about {topic} in 100 words")

#prompt template for llama3
prompt2=ChatPromptTemplate.from_template("write a poem about {topic} in 100 words")

# Add routes to the app
add_routes(app,
           prompt1|llm2,
           path='/essay')

add_routes(app,
           prompt2|llm3,
           path='/poem')

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)
