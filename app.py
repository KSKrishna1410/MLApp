import streamlit as st
import streamlit.components.v1 as components
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain import OpenAI, VectorDBQA
import tempfile
import os
import base64
import fitz
import threading

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

tex_lock = threading.Lock()
text = ""

def extract_text_from_pdf(pdf_path):
    global text
    pdf_document = fitz.open(pdf_path)
    for page_number in range(pdf_document.page_count):
        page = pdf_document[page_number]
        page_text = page.get_text("text", flags=quality)
        with tex_lock:
            text += f'Page {page_number + 1}:\n{page_text}\n'
    pdf_document.close()
    return text

quality = 1

@st.cache  # Cache the text extraction function
def extract_and_process_text(pdf_path, openai_api_key, user_question):
    # Extract text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path)

    # Load the PDF and process the documents
    doc_loader = PDFPlumberLoader(pdf_path)
    documents = doc_loader.load()

    text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1000)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector search
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Chroma.from_documents(texts, embeddings)

    # Initialize the OpenAI model and QA chain
    llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=openai_api_key)
    qa_chain = VectorDBQA.from_chain_type(llm=llm, chain_type='stuff', vectorstore=docsearch)

    # Get the answer for the user's question
    result = qa_chain({'query': user_question}, return_only_outputs=True)

    return extracted_text, result

# Upload PDF and get user's question
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
user_question = st.text_input("Ask a question:")
submit_button = st.button("Submit")

if submit_button and uploaded_file is not None and user_question:
    # Save the uploaded PDF to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        temp_file.write(uploaded_file.read)

    # Extract and process text (this will be cached)
    extracted_text, result = extract_and_process_text(temp_filename, OPENAI_API_KEY, user_question)

    # Display the answer
    if result:
        st.header("Response:")
        st.write(result)
    else:
        st.warning("No answer found for the given question.")
