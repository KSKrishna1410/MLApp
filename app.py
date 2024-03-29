__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import base64
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.chains.question_answering import load_qa_chain, LLMChain
from langchain.llms import AzureOpenAI,OpenAI
from langchain.document_loaders import PyMuPDFLoader as PL
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain import OpenAI, VectorDBQA
from PIL import Image

# Your OpenAI API key
OPENAI_API_KEY = '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'

# Path to the logo image in your repository
logo_image_url = "30188-Sail Analytics-Logo-SH_02 (1).png"

st.set_page_config(page_title='DocuSail by Sail Analytics', page_icon=logo_image_url, initial_sidebar_state='auto')
st.markdown(
    """
    <style>
    .logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: -40px;
        margin-bottom: 30px;
    }
    .logo-text {
        font-weight: 800 !important;
        font-size: 30px !important;
        color: dark !important;
        #padding-top: 20px !important;
    }
    .logo-img {
        width: 100px !important;
        height: 100px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="logo-container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(logo_image_url, "rb").read()).decode()}">
        <p class="logo-text">DocuSail By Sail Analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .css-1rs6os {visibility: hidden;}
            .css-17ziqus {visibility: hidden;}
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Upload PDF and get user's question
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(uploaded_file.read())

    # Load the PDF and process the documents
    doc_loader = PL(temp_filename)
    documents = doc_loader.load()

    text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=10000)
    texts = text_splitter.split_documents(documents)

    template = """Use the following pieces of context to answer the question at the end. It is important to read and understand all the pieces of context provided, as they will help you answer the questions accurately.
If you are unsure about any part of the document, it is best to admit that you don't know and seek clarification. Making up an answer could lead to misunderstandings and legal issues later on. 
and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:
Question: {question}
Helpful Answer:
"""
    template2 = """Question: {question}
Helpful Answer:
Question: {question}
Helpful Answer:
"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    QA_CHAIN_PROMPT2 = PromptTemplate.from_template(template2)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    docsearch = FAISS.from_documents(texts, embeddings)
    llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=OPENAI_API_KEY)
    
    chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, })
   
    
user_question = st.text_input("Ask a question:")
submit_button = st.button("Submit")

if submit_button and user_question:
    # Save the uploaded PDF to a temporary file
    docs = docsearch.similarity_search(user_question)
    # Get the answer for the user's question
    result = chain.run({"query": f'do not include unrelated information and answer must be of 3 lines only :{user_question}',
                        'input_documents': docs, 'return_only_outputs': True}).replace('<|im_end|>','')
    result2 = chain.run({"query": 'Generate 2 questions and helpful answers from given documents', 'input_documents': docs,
                         'return_only_outputs': True, "prompt": QA_CHAIN_PROMPT2})

    # Display the answer
    if result and result2:
        st.header("Response:")
        st.write(result)
        st.write(result2)
    else:
        st.warning("No answer found for the given question.")
