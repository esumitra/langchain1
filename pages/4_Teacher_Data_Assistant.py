# This demo uses LangChain experimental features
# see https://github.com/langchain-ai/langchain/discussions/11680

from typing import Any
import streamlit as st
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import convert_to_secret_str
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from streamlit.logger import get_logger
import configparser 

# global objects
# log: streamlit logger
# config: for application config
# st: for streamlit components
# initialize global objects

# session keys
# api_key = Open AI API key
def init(appname: str) -> None:
  global log, config
  log = get_logger(__name__)
  config = configparser.ConfigParser()
  config.read('app.toml')
  log.info('started %s', appname)

def create_chat_client() -> ChatOpenAI:
  # create chat client with api_key and model name
  model_name:str = config.get('Data_Assistant', 'model_name')
  client_id:str = config.get('Data_Assistant', 'client_name')
  log.info('using model %s', model_name)
  open_ai_key = st.session_state.api_key
  llm = ChatOpenAI(
    name=client_id,
    temperature=0,
    api_key=convert_to_secret_str(open_ai_key),
    model=model_name
  )
  return llm

def load_csv_file(file_path:str) -> pd.DataFrame:
  log.info(f'loading input data file: ${file_path}')
  df = pd.read_csv(file_path)
  with st.expander('See data'):
    tmp_df = df.copy(deep=True)
    st.write(tmp_df)
  return df

@st.cache_resource(max_entries=1)
def create_agent(csv_file:str):
  '''Creates an dataframe agent for input CSV file and caches the result.

  Returns a pandas data frame agent that can be used for querying.
  '''
  log.info(f'creating new agent ...')
  llm = create_chat_client()
  df = load_csv_file(csv_file)
  agent = create_pandas_dataframe_agent(llm, df, verbose=True, agent_type=AgentType.OPENAI_FUNCTIONS)
  return agent

def generate_response(csv_file:str, query: str) -> str:
  agent = create_agent(csv_file)
  response = agent.run(query)
  return response

def page(title:str, desc:str) -> None:
  global openai_api_key, uploaded_file
  st.set_page_config(page_title=title)
  st.title(title)
  st.markdown(desc)
  openai_api_key = st.text_input('OpenAI API Key', type = 'password', key='api_key')
  with st.form('file upload'):
    uploaded_file = st.file_uploader('Upload a CSV data file', type=['csv'])
    query_text = st.text_input('Enter your question:', placeholder = 'How many rows in the file?')
    submitted = st.form_submit_button('Answer')
    if submitted and openai_api_key.startswith('sk-') and uploaded_file is not None:
      with st.spinner('Analyzing file...'):
        result = generate_response(uploaded_file.name, query_text)
        st.info(result)

# main entry point
init('Data Assistant')
page('A Virtual Data Assistant for Teachers',
     '''
     Quickly learn key insights into your students and classes by asking questions about an uploaded data file.

     Sample questions to ask:

     - How many students are performing below grade level?
     - What is the median score of the class?
     - Which schools have English learners with less than 25% mid or above grade level?
     ''')
