import streamlit as st
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from streamlit.logger import get_logger
import configparser 

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

def generate_response(text: str) -> str:
  model_name:str = config.get('teacher_topic_summary', 'model_name')
  client_id:str = config.get('teacher_topic_summary', 'client_name')
  log.info('using model %s', model_name)
  # instantiate llm model
  llm = OpenAI(temperature=0.0, openai_api_key=openai_api_key, model=model_name, client=client_id)
  # split text
  text_splitter = CharacterTextSplitter()
  txts = text_splitter.split_text (text)
  # create multiple documents
  docs = [Document(page_content=t) for t in txts]
  # summarize text
  chain = load_summarize_chain(llm, chain_type='map_reduce')
  result = chain.run(docs)
  return result


def page(title:str, desc:str) -> None:
  st.set_page_config(page_title=title)
  st.title(title)
  st.write(desc)
  global openai_api_key
  result = []
  with st.form('summarize_form', clear_on_submit=False):
    openai_api_key = st.text_input('OpenAI API Key', type = 'password')
    txt_input = st.text_area('Enter your text', '', height=200)
    submitted = st.form_submit_button('Submit')
    if (submitted and openai_api_key.startswith('sk-')):
      with st.spinner('Calculating summary...'):
        response = generate_response(txt_input)
        result.append(response)
    if len(result):
      st.info(result)

# main entry point
init('Teacher Summary')
page('Teacher Topic Summary', 'A teacher tool to improve student engagement by summarizing current scientific articles.')
