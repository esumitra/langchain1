from typing import Any
import streamlit as st
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
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

def load_csv_file(file_path:str) -> pd.DataFrame:
  log.info(f'loading input data file: ${file_path}')
  df = pd.read_csv(file_path)
  with st.expander('See data'):
    tmp_df = df.copy(deep=True)
    st.write(tmp_df)
  return df

@st.cache_resource(max_entries=1)
def create_agent(csv_file:str):
  '''Creates an agent for a file and caches the result'''
  log.info(f'creating new agent ...')
  global config
  model_name:str = config.get('Data_Assistant', 'model_name')
  client_id:str = config.get('Data_Assistant', 'client_name')
  temp = config.get('Data_Assistant', 'temperature')
  llm = ChatOpenAI(model=model_name, temperature=float(temp), openai_api_key=openai_api_key)
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
  openai_api_key = st.text_input('OpenAI API Key', type = 'password')
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
     - Which schools have Yes English learners with less than 25% mid or above grade level?
     ''')
