# Import necessary modules
import re
import time
from io import BytesIO
from typing import Any, Dict, List
from PIL import Image
import openai
import streamlit as st
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
from langchain.retrievers.merger_retriever import MergerRetriever
import os
from dotenv import load_dotenv

load_dotenv()
#os.environ['OPENAI_API_KEY'] = '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'

# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output


# Define a function to convert text content to a list of documents
@st.cache_resource
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


# Define a function for the embeddings
@st.cache_resource
def test_embed():
    embeddings = OpenAIEmbeddings()
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Embeddings done.", icon="✅")
    return index


# Set up the Streamlit app
st.title("RAG Chatbot for mutiple documents with Conversational Memory")

st.markdown(
    """
    `openai`
    `langchain`
    `tiktoken`
    `pypdf`
    `faiss-cpu`
    ---------
    """
)

# Set up the sidebar
st.sidebar.markdown(
    """
    ### Steps:
    1. Upload PDF File
    3. Perform Q&A
    """
)
image = Image.open('1_62A8TARBY4WTYt5RJR7PKA.png')
st.sidebar.image(image,caption="Working of the chatbot",use_column_width=True)

# Allow the user to upload a PDF file
uploaded_files = st.file_uploader("**Upload Your PDF File**", type=["pdf"],accept_multiple_files=True)
name_of_file = []
pages = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        name_of_file.append(uploaded_file.name)
        doc = parse_pdf(uploaded_file)
        pages = pages + text_to_docs(doc)
        if pages:
            # Allow the user to select a page and view its content
            with st.expander("Show Page Content", expanded=False):
                page_sel = st.number_input(
                    label="Select Page", min_value=1, max_value=len(pages), step=1
                )
                pages[page_sel - 1]
            # # Allow the user to enter an OpenAI API key
            # api = st.text_input(
            #     "**Enter OpenAI API Key**",
            #     type="password",
            #     placeholder="sk-",
            #     help="https://platform.openai.com/account/api-keys",
            # )
        
        # Test the embeddings and save the index in a vector database
                
    index = test_embed()
    #lotr = MergerRetriever(retrievers=[i.as_retriever() for i in index])
    # Set up the question-answering system
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type = "stuff",
        retriever=index.as_retriever()
    )
    # Set up the conversational agent
    tools = [
        Tool(
            name="State of Union QA System",
            func=qa.run,
            description="Useful for when you need to answer questions about the aspects asked. Input may be a partial or fully formed question.",
        )
    ]
    prefix = """Have a conversation with a human, answering the following questions as best you can based on the context and memory available. 
                You have access to a single tool:"""
    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history"
        )

    llm_chain = LLMChain(
        llm=OpenAI(
            temperature=0, model_name="gpt-3.5-turbo"
        ),
        prompt=prompt,
    )
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=st.session_state.memory,handle_parsing_errors=True
    )

    # Allow the user to enter a query and generate a response
    query = st.text_input(
        "**What's on your mind?**",
        placeholder="Ask me anything from {}".format(",".join(name_of_file)),
    )

    if query:
        with st.spinner(
            "Generating Answer to your Query : `{}` ".format(query)
        ):
            res = agent_chain.run(query)
            st.info(res, icon="🤖")
            st.write(f"Document Used by Chatbot: {index.as_retriever().get_relevant_documents(query)}")
    # Allow the user to view the conversation history and other information stored in the agent's memory
    with st.expander("History/Memory"):
        st.session_state.memory
