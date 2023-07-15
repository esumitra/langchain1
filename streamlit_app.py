import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
from streamlit.logger import get_logger

log = get_logger(__name__)

log.info('starting app')

st.title('AI-Den, Your Algebra Study Buddy')

open_ai_key:str = st.sidebar.text_input('OpenAI API Key')

def generate_response(topic:str) -> str:
  llm = OpenAI(temperature=0.7, openai_api_key=open_ai_key, model="text-davinci-003", client="a")
  template = 'As an experienced middle school teacher explain with detailed steps the algebra topic {topic}.'
  prompt = PromptTemplate(input_variables=['topic'], template=template)
  prompt_query = prompt.format(topic = topic)
  log.info('calling OpenAPI with text %s', prompt_query)
  response: str = llm(prompt_query)
  return response
  

with st.form('my_form'):
  text = st.text_area('What topic would you like to learn about?', value='solving equations')
  submitted = st.form_submit_button('Submit')
  if not open_ai_key.startswith('sk-'):
    st.warning('Please enter your Open API key', icon='⚠')
  if submitted and open_ai_key.startswith('sk-'):
    res:str = generate_response(text)
    st.info(res)



