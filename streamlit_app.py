import streamlit as st
from langchain.llms import OpenAI

st.title('Ed\'s Quickstart App')

open_ai_key:str = st.sidebar.text_input('OpenAI API Key')

def generate_response(input_text:str) -> None:
  llm = OpenAI(temperature=0.7, openai_api_key=open_ai_key, model="text-davinci-003", client="a")
  st.info(llm(input_text))

with st.form('my_form'):
  text = st.text_area('Enter your question here', value='Explain how to solve a quadratic equation to a middle school student with steps')
  submitted = st.form_submit_button('Submit')
  if not open_ai_key.startswith('sk-'):
    st.warning('Please enter your Open API key', icon='⚠')
  if submitted and open_ai_key.startswith('sk-'):
    generate_response(text)



