## AI Algebra Study Buddy
An AI Algebra Study Buddy that can help school students with their math practice. The buddy is built with Chat-GPT and LangChain.


### Install developer environment
1. Create a local environment for the project
e.g.,
```
conda create -n chatgpt1 python=3.9
conda use chatgpt1
```

2. Install the dependencies

```
echo -e "streamlit\nopenai\nlangchain" > requirements.txt
pip install -r requirements.txt
```

3. Run the app locally

`streamlit run streamlit_app.py`

### Deploy to Streamlit Cloud
To deploy to streamlit cloud
1. Create a public repo on github
2. Connect streamlit could to the github account in #1
3. Create a new streamlit app from the github repo and main branch
4. Push changes to the main branch

### License
Copyright 2023, esumitra

Licensed under the MIT License.