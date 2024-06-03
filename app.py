import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template

def get_pdf_text(pdf_docs):
    texts = []
    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        texts.append(text)
    return texts

def get_vectorstore(texts):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    return vectorstore

def get_summary_chain(vectorstore):
    llm = ChatOpenAI()
    summary_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=None
    )
    return summary_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Process and Compare Vendors quotes from  PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header(" Compare Vendors quotes from  PDFs")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        
    if st.sidebar.button("Process"):
        with st.spinner("Processing"):
            # get pdf text
            texts = get_pdf_text(pdf_docs)

            # create vector store
            vectorstore = get_vectorstore(texts)

            # create summary chain
            summary_chain = get_summary_chain(vectorstore)
            
            # get summary
            response = summary_chain({'question':  """For each document, specify the details fetched. and perfom a comparison between them""", 'chat_history': []})
            
            summary = response['answer']

            st.write(bot_template.replace(
                "{{MSG}}", summary), unsafe_allow_html=True)

if __name__ == '__main__':
    main()
