from typing import Any
import streamlit as st
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.utils.utils import convert_to_secret_str
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from streamlit.logger import get_logger
import configparser 
from dataclasses import dataclass

# global objects
# log: streamlit logger
# config: for application config
# st: for streamlit components

# session keys
# api_key = Open AI API key

def init(appname: str) -> None:
  global log, config
  log = get_logger(__name__)
  config = configparser.ConfigParser()
  config.read('app.toml')
  log.info('started %s', appname)

def create_chat_client() -> ChatOpenAI:
  """Create a chat client with model name and api key
  """
  model_name:str = config.get('openai', 'model_name')
  client_id:str = config.get('openai', 'client_name')
  log.info('using model %s', model_name)
  open_ai_key = st.session_state.api_key
  llm = ChatOpenAI(
    name=client_id,
    temperature=0,
    api_key=convert_to_secret_str(open_ai_key),
    model=model_name
  )
  return llm

# TODO: needs error handling
@st.cache_resource(max_entries=5)
def handle_upload(name:str, content: list[str]) -> BaseRetrievalQA:
  """Chunks text from content to create embeddings that are stored
  in a vector database.
  
  Returns an chain that can be used for Q and A with the content.
  """
  global config
  client_id:str = config.get('Q_and_A', 'client_name')
  model_name:str = config.get('Q_and_A', 'model_name') # needs an embeddings model
  openai_api_key:str = st.session_state.api_key

  # Load document if file is uploaded
  documents = content
  # Split documents into chunks
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.create_documents(documents)

  # Select embeddings
  log.info(f'creating embeddings for file: {name}')
  embeddings = OpenAIEmbeddings(api_key=convert_to_secret_str(openai_api_key), model=model_name)

  # Create a vectorstore from documents
  db = Chroma.from_documents(texts, embeddings)
  # Create retriever interface
  retriever = db.as_retriever()

  # Create QA chain
  llm = create_chat_client()
  qa  = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever = retriever)
  return qa

def page(title:str, desc:str) -> None:
  global openai_api_key
  st.set_page_config(page_title=title)
  st.title(title)
  st.markdown(desc)
  openai_api_key = st.text_input('OpenAI API Key', type = 'password', key='api_key')
  with st.form('file upload'):
    uploaded_file = st.file_uploader('Upload an article', type='txt')
    query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.')
    submitted = st.form_submit_button('Analyze')
    if submitted and openai_api_key.startswith('sk-') and uploaded_file is not None:
      with st.spinner('Analyzing file...'):
        docs = [uploaded_file.read().decode()]
        qa = handle_upload(uploaded_file.name, docs)
        result = qa.run(query_text)
        st.info(result)

# main entry point
init('Comprehension QnA')
page('An AI History Teacher',
     '''
     Learn more about important historical events and topics through famous speeches. 
     Select or upload a famous speech and ask questions about the speech.

     For inspiration, check out the speeches at [American Rhetoric](https://www.americanrhetoric.com/top100speechesall.html)
     ''')
