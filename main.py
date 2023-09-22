from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
load_dotenv()
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import streamlit as st
import pandas as pd
import os, tempfile

#제목
st.title("ChatPDF")
st.write('---')

uploaded_file = st.file_uploader('Choose a file')
st.write('---')

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory() 
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, 'wb') as f:
      f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
#Split
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size = 300,
        chunk_overlap = 20,
        length_function = len,
        add_start_index = True,
    )

    #Embedding
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings()

    #load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header('질문하세요!!')
    question = st.text_input('질문을 입력하세요.')

    if st.button('질문하기'):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
        result = qa_chain({"query": question})
        st.write(result['result'])