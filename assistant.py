import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
import chromadb.api

chromadb.api.client.SharedSystemClient.clear_system_cache()
from dotenv import load_dotenv
load_dotenv()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title("Business Guidance ChatBot")


loader = PyPDFLoader("NIC_merged.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)


embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=os.getenv("OPENAI_API_KEY"))

prompt_template = """
you are a businees formation assistant
You are an assistant and you have two jobs , first is to give NIC code for the business detail which the user gives based on the pdf.
and second based on the nic code give them full guide on how to set up that bussiness in india.
and if the user not ask for nic code just give them the business setting up advice and guide in india based on the pdf.
and if the user ask about what type of registration they should form (like LLP , PVT ,sole propriership etc) give them the answer based on there business type and need and if you don't know ask user about there business and give proper answer.
Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you 
don't know. Use three sentences maximum and keep the answer perfectly and give exact NIC code for the business and perfect step by step guide for setting up the business.

Previous conversation:
{chat_history}

users question or response: {input}

Use the following context for nic code and Doing Business in India:
{context}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Create chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

with st.chat_message("ai"):
    st.markdown("Hello! I am your business guidance assistant. I can help you find NIC codes for businesses and provide detailed guidance on setting up businesses in India. How can I assist you today?")



for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.markdown(message.content)
    else:
        with st.chat_message("ai"):
            st.markdown(message.content)


if query := st.chat_input("Your message:"):
    with st.chat_message("human"):
        st.markdown(query)
    
    st.session_state.chat_history.append(HumanMessage(content=query))
    
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            formatted_chat_history = [
                f"{'Human' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content}" 
                for msg in st.session_state.chat_history
            ]
            
            response = rag_chain.invoke({
                "input": query,
                "chat_history": "\n".join(formatted_chat_history)
            })
            
            st.markdown(response["answer"])
            st.session_state.chat_history.append(AIMessage(content=response["answer"]))