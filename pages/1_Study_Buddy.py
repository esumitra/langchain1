import streamlit as st
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.utils.utils import convert_to_secret_str
from streamlit.logger import get_logger
import configparser

# global objects
# log: streamlit logger
# config: for application config
# st: for streamlit components

# session states
# person = personality selected
# api_key = Open AI API key

# initialize global objects
def init(appname: str) -> None:
  global log, config
  log = get_logger(__name__)
  config = configparser.ConfigParser()
  config.read('app.toml')
  log.info('started %s', appname)

# st.session_state.person=''


def get_template_for_person(person: str) -> str:
  """Helper function to load person template from config
  """
  person_key:str = config.get('personality_names', person)
  template:str = config.get('personality_templates', person_key)
  log.debug('using personality template: %s', template)
  return template

def generate_chat_messages(topic:str, person:str) -> List[BaseMessage]:
  """returns a list of system and human messages using prompt template
  from configuration
  """
  # create chat messages from templates
  system_message_prompt_template = 'You are a helpful assistant that helps students with their homework.'
  system_message_prompt = SystemMessagePromptTemplate.from_template(system_message_prompt_template)
  human_message_prompt = HumanMessagePromptTemplate.from_template(get_template_for_person(person))
  chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
  )
  formatted_messages = chat_prompt \
    .format_prompt(topic=topic) \
    .to_messages()
  log.info('using messages\n%s', formatted_messages)
  return formatted_messages

def create_chat_client() -> ChatOpenAI:
  # create chat client with api_key and model name
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


def generate_response(topic:str, person:str) -> str:
  """generates response by calling chat api
  """
  llm = create_chat_client()
  messages = generate_chat_messages(topic, person)
  # get a chat completion from the formatted messages
  result = llm(messages)
  return result.content

def page() -> None:
  # title section
  st.title('Your Algebra Study Buddy')
  st.markdown(
    """
      How would you like a study buddy with a quirky personality? 
      Pick a personality, ask a question, and see how your buddy can help you learn algebra. 💯
    """
  )

  # side bar section for API key and buddy selection
  with st.sidebar:
    open_ai_key:str = st.text_input('OpenAI API Key', type = 'password', key='api_key')
    pnames:list[str] = [x[0] for x in config.items('personality_names')]
    person_name = st.selectbox(
      'What personality would you like for your buddy?',
      pnames,
      key='person', # store value in sesson state with key = 'person'
    )

  # main Q & A form
  with st.form('my_form'):
    text = st.text_area('What topic would you like to learn about?', value='solve 5x + 3y = 3')
    submitted = st.form_submit_button('Submit')
    if not open_ai_key.startswith('sk-'):
      st.warning('Please enter your Open API key', icon='⚠')
    if submitted and open_ai_key.startswith('sk-'):
      person = st.session_state.person
      res:str = generate_response(text, person)
      st.info(res)

# main entry function
init("study buddy")
page()
