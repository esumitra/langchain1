from typing import Any
import streamlit as st
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from streamlit.logger import get_logger
import configparser 
from dataclasses import dataclass

# global objects
# log: streamlit logger
# config: for application config
# st: for streamlit components
# initialize global objects
openai_api_key:str=''

def init(appname: str) -> None:
  global log, config
  log = get_logger(__name__)
  config = configparser.ConfigParser()
  config.read('app.toml')
  log.info('started %s', appname)

@st.cache_resource(max_entries=5)
def handle_upload(name:str, content: list[str]) -> BaseRetrievalQA:
  '''generates model resources when a new file is uploaded
  model resources are stored in a global variable qadb'''
  log.info(f'creating embedding and vector store for file: {name}')
  global uploaded_file, config
  model_name:str = config.get('Q_and_A', 'model_name')
  client_id:str = config.get('Q_and_A', 'client_name')

  # Load document if file is uploaded
  documents = content
  # Split documents into chunks
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.create_documents(documents)

  # Select embeddings
  embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key,client=client_id)

  # Create a vectorstore from documents
  db = Chroma.from_documents(texts, embeddings)

  # Create retriever interface
  retriever = db.as_retriever()

  # Create QA chain
  llm = OpenAI(openai_api_key=openai_api_key, model=model_name, client=client_id)
  qa  = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever = retriever)
  return qa

def page(title:str, desc:str) -> None:
  global openai_api_key, uploaded_file
  #qa = ''
  st.set_page_config(page_title=title)
  st.title(title)
  st.markdown(desc)
  openai_api_key = st.text_input('OpenAI API Key', type = 'password')
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
