import streamlit as st
from langchain.llms import OpenAI
from langchain import PromptTemplate
from streamlit.logger import get_logger
import configparser 

# global objects
# log: streamlit logger
# config: for application config
# st: for streamlit components

# initialize global objects
log = get_logger(__name__)

log.info('starting app')
config = configparser.ConfigParser()
config.read('app.toml')
# st.session_state.person=''
st.title('Your Algebra Study Buddy')
st.markdown(
  """
    How would you like a study buddy with a quirky personality? 
    Pick a personality, ask a question, and see how your buddy can help you learn algebra. 💯
  """
)
with st.sidebar:
  open_ai_key:str = st.text_input('OpenAI API Key')
  pnames:list[str] = [x[0] for x in config.items('personality_names')]
  person_name = st.selectbox(
    'What personality would you like for your buddy?',
    pnames,
    key='person', # store value in sesson state with key = 'person'
  )

def get_template_for_person(person: str) -> str:
  person_key:str = config.get('personality_names', person)
  template: str = config.get('personality_templates', person_key)
  log.debug('using personality template: %s', template)
  return template

def generate_response(topic:str, person:str) -> str:
  model_name:str = config.get('openai', 'model_name')
  client_id:str = config.get('openai', 'client_name')
  log.info('using model %s', model_name)
  llm = OpenAI(temperature=0.7, openai_api_key=open_ai_key, model=model_name, client=client_id)
  template = get_template_for_person(person)
  prompt = PromptTemplate(input_variables=['topic'], template=template)
  prompt_query = prompt.format(topic = topic)
  log.info('calling OpenAPI with text %s', prompt_query)
  response: str = llm(prompt_query)
  return response

with st.form('my_form'):
  text = st.text_area('What topic would you like to learn about?', value='solve 5x + 3y = 3')
  submitted = st.form_submit_button('Submit')
  if not open_ai_key.startswith('sk-'):
    st.warning('Please enter your Open API key', icon='⚠')
  if submitted and open_ai_key.startswith('sk-'):
    person = st.session_state.person
    res:str = generate_response(text, person)
    st.info(res)



