import streamlit as st
#cache_resource
@st.cache_data
def mymain():
    import os
    from apiKey import OPENAI_API_KEY
    
    from langchain.llms import OpenAI

    os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

    from PyPDF2 import PdfReader
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS


    reader = PdfReader('django.pdf')

    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text


    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)


    from langchain.chains.question_answering import load_qa_chain
    from langchain.llms import OpenAI

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    st.title("DjangoGPT")
    return chain,docsearch

chain,docsearch = mymain()

prompt = st.text_input("ask about django")
if prompt:
    docs = docsearch.similarity_search(prompt)
    response = chain.run(input_documents=docs, question=prompt)
    st.write(response)