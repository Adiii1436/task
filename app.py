import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains.question_answering import load_qa_chain
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import uuid
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import tempfile

def process_uploaded_files(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        try:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            if file_ext == '.txt' or file_ext == '.md':
                loader = TextLoader(tmp_path)
            elif file_ext == '.pdf':
                loader = UnstructuredPDFLoader(tmp_path)
            elif file_ext == '.docx':
                loader = UnstructuredWordDocumentLoader(tmp_path)
            else:
                st.sidebar.warning(f"Unsupported file type: {file_ext}")
                continue
            
            documents.extend(loader.load())
            os.unlink(tmp_path)
        except Exception as e:
            st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    return documents

def create_vector_db(documents):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    
    if documents:
        docs = text_splitter.split_documents(documents)
        vectordb = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory="chroma_db"
        )
        vectordb.persist()
        return vectordb
    return None

# Streamlit UI setup
st.set_page_config(page_title="Datadoc", page_icon=":robot:", layout="wide", initial_sidebar_state="expanded")

col1, col2 = st.columns([6,1])
with col1:
    st.title("Datadoc: Your AI DOC Assistant")

# Initialize session state
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'vectordb' not in st.session_state:
    st.session_state.vectordb = None

# Sidebar for options and file upload
st.sidebar.title("Options")

model_name = st.sidebar.radio("Choose a model:", ("Gemini 2.0 Flash"))
api_key = st.sidebar.text_input("Enter your API key:", type="password")

# File upload section in sidebar
uploaded_files = st.sidebar.file_uploader(
    "Upload text files to add to knowledge base",
    type=["txt", "pdf", "docx", "md"],
    accept_multiple_files=True
)

if uploaded_files and api_key:
    documents = process_uploaded_files(uploaded_files)
    if documents:
        st.session_state.vectordb = create_vector_db(documents)
        st.sidebar.success(f"Processed {len(documents)} documents and created new knowledge base!")
    else:
        st.sidebar.warning("No valid documents were processed")

clear_button = st.sidebar.button("Clear Conversation", key="clear")

if model_name == "Gemini 2.0 Flash":
    model = "gemini-2.0-flash"

if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []

def generate_response(prompt, model, image_path=None, explain_to_kid=False):
    if len(st.session_state['past']) == len(st.session_state['generated']):
        st.session_state['past'].append(prompt)
    else:
        st.session_state['past'][-1] = prompt

    os.environ["GOOGLE_API_KEY"] = api_key
    chat_history = []
    
    if st.session_state.vectordb is None:
        st.warning("Please upload some documents first!")
        return "No documents available to answer questions."

    response = ""
    
    text_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    retriever = st.session_state.vectordb.as_retriever()
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | text_llm | StrOutputParser()
    if explain_to_kid:
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        You're an experienced teacher who loves simplifying complex topics for young children to understand. \
        Your task is to explain a complex topic as if you are talking to a 5-year-old. \
        Make sure to use playful and engaging language to keep the child's attention and break down any difficult ideas into simple, manageable parts.For example, if you were explaining photosynthesis, you could say something like: "Plants eat sunlight by dancing in the sun all day long. Imagine the sun as their yummy snack! They also drink from the ground through their roots like using a straw. With these snacks and drinks, they make their own food just like magic!" \
        If you don't know the answer, just say that you don't know. \
        {context}"""
    else:
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        Try to analyze the given context and answer the question to the best of your ability. \
        If you don't know the answer, just say that you don't know. \
        {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | text_llm
    )
    res = rag_chain.invoke({"question": prompt, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=prompt), res])
    response = res.content

    if len(st.session_state['generated']) < len(st.session_state['past']):
        st.session_state['generated'].append(response)
    else:
        st.session_state['generated'][-1] = response

    return response

# Main chat interface
response_container = st.container()
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100, max_chars=500)
        
        uploaded_file = None
        if model_name == "Gemini Pro Vision":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns([6, 1])
        with col1:
            submit_button = st.form_submit_button(label='Send')
        with col2:
            explain_kid = st.toggle("Child Mode", key='explain_toggle')
        
    if submit_button and not api_key:
        st.warning("Please enter your API key.")
    elif submit_button and not uploaded_file and model_name == "Gemini Pro Vision":
        st.warning("Please upload an image to use the Image Model.")
    elif submit_button and uploaded_file and model_name == "Gemini Pro Vision":
        image_path = save_uploaded_image(uploaded_file)
        if image_path:
            output = generate_response(user_input, model, image_path, explain_kid)
    elif submit_button and user_input:
        output = generate_response(user_input, model, explain_to_kid=explain_kid)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            st.markdown(f"**You:** {st.session_state['past'][i]}")
            with st.container(border=True):
                st.markdown(f"{st.session_state['generated'][i]}")