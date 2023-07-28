import streamlit as st

def home():
    import streamlit as st

    st.write("# Welcome to Ai Tools for the Classroom! 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Explore proof of concept demos on how AI can help students and teachers in the classroom.
        You will need an OpenAI API key from https://openai.com/ to run the demos.

        **👈 Select a demo from links the left** to see examples
        of what AI can do for students and teachers!

        
        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io) for the UI Platform used to build the web application
        - Check out [LangChain](https://python.langchain.com/docs/get_started/introduction.html) a framework for developing Large Language Model (LLM) application.

    """
    )

# main entry point
home()
