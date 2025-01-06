# import os
# import logging
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from langchain import hub
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.schema import Document
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/test-cors")
# async def test_cors():
#     return {"message": "CORS works!"}

# class Question(BaseModel):
#     query: str
#     session_id: str = "default"

# loader = PyMuPDFLoader('fam Swissmart.pdf'),

# # docs = [doc for loader in loaders for doc in loader.load()]
# docs = loader[0].load()

# # Define the headers
# headers = [
#     "RELATIONSHIP SUMMARY",
#     "GROUP FACILITY SUMMARY",
#     "GROUP EXPOSURE SUMMARY",
#     "SECURITY/SUPPORT STRUCTURE",
#     "SECURITY COVERAGE ANALYSIS",
#     "SIGNIFICANT EXPOSURE DETAILS",
#     "BACKGROUND TO THE REQUEST",
#     "APPROVAL REQUEST",
#     "CRITICAL CREDIT ISSUES",
#     "RISK/RETURN AND RELATIONSHIP STRATEGY",
#     "CONDITIONS PRECEDENT TO DRAWDOWN",
#     "TRANSACTION DYNAMICS",
#     "WAYS OUT ANALYSIS",
#     "BANKING RELATIONSHIPS AND EXISTING FACILITIES",
#     "RISK ANALYSIS",
#     "COUNTRY OVERVIEW",
#     "OWNERSHIP AND MANAGEMENT ASSESSMENT",
#     "HISTORICAL FINANCIAL ANALYSIS & OUTLOOK",
#     "PROJECTED FINANCIAL ANALYSIS",
#     "FINANCIAL PROJECTION",
#     "BUSINESS AND INDUSTRY DESCRIPTION",
#     "CORPORATE STRUCTURE AND ORGANIZATION",
#     "JUSTIFICATION FOR THE REQUEST/CREDIT CONSIDERATION",
#     "APPROVAL RECOMMENDATION",
# ]

# # Create a regular expression pattern to match the headings
# header_pattern = re.compile(rf"\b({'|'.join(map(re.escape, headers))})\b")

# # Custom function to split documents by headings
# def split_by_headings(document_text: str, headers: list):
#     # Find all matches for headings in the text
#     matches = list(header_pattern.finditer(document_text))
#     chunks = []

#     for i, match in enumerate(matches):
#         start = match.start()
#         end = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
#         heading = match.group(0).strip()
#         content = document_text[start:end].strip()
#         chunks.append({"heading": heading, "content": content})

#     return chunks

# document_text = " ".join(doc.page_content for doc in docs)
# # Split the document into chunks based on headings
# chunks = split_by_headings(document_text, headers)
# print(chunks, 'these are chunks')

# # Convert chunks into Document objects
# splits = [
#     Document(page_content=chunk["content"], metadata={"heading": chunk["heading"]})
#     for chunk in chunks
# ]
# print(splits, 'these are splits')

# # Create the vectorstore
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# # Retrieve and generate using the relevant snippets of the document
# retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# print(len(splits))

# logging.info("Created vectorstore")

# # 4. Retrieve and generate using the relevant snippets of the blog.
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# also when you are asked what you are trained on or which document do you contain say you were trained on the prudential guidelines, Access bank's CPG and the Basel Frameworks, note that this quesion can come in different ways \
# If you don't know the answer or if the answer is not in the document you are trained on, just say that the answer is not in the content provided and do not answer the question . \
# ALWAYS provide the reference document(s) for your answer.

# just reformulate it if needed and otherwise return it as is."""
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

# qa_system_prompt = """You are a financial analysis assistant specializing in Facility Approval Memos (FAMs). \
# Your task is to provide a detailed and comprehensive evaluation of the FAM based on the provided context. \
# Your responses must be thorough, precise, and grounded in the information from the FAM. If the required information is not present in the context, explicitly state that it is missing and suggest what additional information would be needed.

# For every question or request:
# - Provide a detailed explanation that addresses all aspects of the question.
# - Reference the relevant section(s) of the FAM to support your response.
# - Highlight any gaps, inconsistencies, or missing details in the provided context.
# - Offer actionable recommendations or insights where applicable.

# Ensure your analysis is professional, well-structured, and adheres to financial analysis standards.

# {context}"""
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

# @app.post("/ask")
# async def ask_question(question: Question):
#     try:
#         result = conversational_rag_chain.invoke(
#             {"input": question.query},
#             config={"configurable": {"session_id": question.session_id}}
#         )

#         get_session_history(question.session_id).add_user_message(question.query)
#         get_session_history(question.session_id).add_ai_message(result['answer'])
#         return {"answer": result['answer']}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=8000)

# import os
# import logging
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from langchain import hub
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.document_loaders import DirectoryLoader
# from langchain.schema import Document
# import re

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.get("/test-cors")
# async def test_cors():
#     return {"message": "CORS works!"}

# class Question(BaseModel):
#     query: str
#     session_id: str = "default"

# loader = PyMuPDFLoader('fam Swissmart.pdf'),

# # docs = [doc for loader in loaders for doc in loader.load()]
# docs = loader[0].load()

# # Define the headers
# headers = [
#     "RELATIONSHIP SUMMARY",
#     "GROUP FACILITY SUMMARY",
#     "GROUP EXPOSURE SUMMARY",
#     "SECURITY/SUPPORT STRUCTURE",
#     "SECURITY COVERAGE ANALYSIS",
#     "SIGNIFICANT EXPOSURE DETAILS",
#     "BACKGROUND TO THE REQUEST",
#     "APPROVAL REQUEST",
#     "CRITICAL CREDIT ISSUES",
#     "RISK/RETURN AND RELATIONSHIP STRATEGY",
#     "CONDITIONS PRECEDENT TO DRAWDOWN",
#     "TRANSACTION DYNAMICS",
#     "WAYS OUT ANALYSIS",
#     "BANKING RELATIONSHIPS AND EXISTING FACILITIES",
#     "RISK ANALYSIS",
#     "COUNTRY OVERVIEW",
#     "OWNERSHIP AND MANAGEMENT ASSESSMENT",
#     "HISTORICAL FINANCIAL ANALYSIS & OUTLOOK",
#     "PROJECTED FINANCIAL ANALYSIS",
#     "FINANCIAL PROJECTION",
#     "BUSINESS AND INDUSTRY DESCRIPTION",
#     "CORPORATE STRUCTURE AND ORGANIZATION",
#     "JUSTIFICATION FOR THE REQUEST/CREDIT CONSIDERATION",
#     "APPROVAL RECOMMENDATION",
# ]

# # Create a regular expression pattern to match the headings
# header_pattern = re.compile(rf"\b({'|'.join(map(re.escape, headers))})\b")

# # Custom function to split documents by headings
# def split_by_headings(document_text: str, headers: list):
#     # Find all matches for headings in the text
#     matches = list(header_pattern.finditer(document_text))
#     chunks = []

#     for i, match in enumerate(matches):
#         start = match.start()
#         end = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
#         heading = match.group(0).strip()
#         content = document_text[start:end].strip()
#         chunks.append({"heading": heading, "content": content})

#     return chunks

# document_text = " ".join(doc.page_content for doc in docs)
# # Split the document into chunks based on headings
# chunks = split_by_headings(document_text, headers)
# print(chunks, 'these are chunks')

# # Convert chunks into Document objects
# splits = [
#     Document(page_content=chunk["content"], metadata={"heading": chunk["heading"]})
#     for chunk in chunks
# ]
# print(splits, 'these are splits')

# # Create the vectorstore
# vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# # Retrieve and generate using the relevant snippets of the document
# retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# print(len(splits))

# logging.info("Created vectorstore")

# # 4. Retrieve and generate using the relevant snippets of the blog.
# contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# also when you are asked what you are trained on or which document do you contain say you were trained on the prudential guidelines, Access bank's CPG and the Basel Frameworks, note that this quesion can come in different ways \
# If you don't know the answer or if the answer is not in the document you are trained on, just say that the answer is not in the content provided and do not answer the question . \
# ALWAYS provide the reference document(s) for your answer.

# just reformulate it if needed and otherwise return it as is."""
# contextualize_q_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", contextualize_q_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# history_aware_retriever = create_history_aware_retriever(
#     llm, retriever, contextualize_q_prompt
# )

# qa_system_prompt = """You are a financial analysis assistant specializing in Facility Approval Memos (FAMs). \
# Your task is to provide a detailed and comprehensive evaluation of the FAM based on the provided context. \
# Your responses must be thorough, precise, and grounded in the information from the FAM. If the required information is not present in the context, explicitly state that it is missing and suggest what additional information would be needed.

# For every question or request:
# - Provide a detailed explanation that addresses all aspects of the question.
# - Reference the relevant section(s) of the FAM to support your response.
# - Highlight any gaps, inconsistencies, or missing details in the provided context.
# - Offer actionable recommendations or insights where applicable.

# Ensure your analysis is professional, well-structured, and adheres to financial analysis standards.

# {context}"""
# qa_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", qa_system_prompt),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#     ]
# )
# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# store = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     if session_id not in store:
#         store[session_id] = ChatMessageHistory()
#     return store[session_id]

# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

# @app.post("/ask")
# async def ask_question(question: Question):
#     try:
#         result = conversational_rag_chain.invoke(
#             {"input": question.query},
#             config={"configurable": {"session_id": question.session_id}}
#         )

#         get_session_history(question.session_id).add_user_message(question.query)
#         get_session_history(question.session_id).add_ai_message(result['answer'])
#         return {"answer": result['answer']}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=8000)
















import os
import logging
import re
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document
from tempfile import NamedTemporaryFile

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/test-cors")
async def test_cors():
    return {"message": "CORS works!"}


class Question(BaseModel):
    query: str
    session_id: str = "default"


# Define the headers
headers = [
    "RELATIONSHIP SUMMARY",
    "GROUP FACILITY SUMMARY",
    "GROUP EXPOSURE SUMMARY",
    "SECURITY/SUPPORT STRUCTURE",
    "SECURITY COVERAGE ANALYSIS",
    "SIGNIFICANT EXPOSURE DETAILS",
    "BACKGROUND TO THE REQUEST",
    "APPROVAL REQUEST",
    "CRITICAL CREDIT ISSUES",
    "RISK/RETURN AND RELATIONSHIP STRATEGY",
    "CONDITIONS PRECEDENT TO DRAWDOWN",
    "TRANSACTION DYNAMICS",
    "WAYS OUT ANALYSIS",
    "BANKING RELATIONSHIPS AND EXISTING FACILITIES",
    "RISK ANALYSIS",
    "COUNTRY OVERVIEW",
    "OWNERSHIP AND MANAGEMENT ASSESSMENT",
    "HISTORICAL FINANCIAL ANALYSIS & OUTLOOK",
    "PROJECTED FINANCIAL ANALYSIS",
    "FINANCIAL PROJECTION",
    "BUSINESS AND INDUSTRY DESCRIPTION",
    "CORPORATE STRUCTURE AND ORGANIZATION",
    "JUSTIFICATION FOR THE REQUEST/CREDIT CONSIDERATION",
    "APPROVAL RECOMMENDATION",
]

# Create a regular expression pattern to match the headings
header_pattern = re.compile(rf"\b({'|'.join(map(re.escape, headers))})\b")


def split_by_headings(document_text: str, headers: list):
    matches = list(header_pattern.finditer(document_text))
    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(
            document_text)
        heading = match.group(0).strip()
        content = document_text[start:end].strip()
        chunks.append({"heading": heading, "content": content})

    return chunks


store = {}
vectorstore = None


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# @app.post("/upload-pdf")
# async def upload_pdf(file: UploadFile = File(...)):
#     global vectorstore

#     try:
#         # Save the uploaded file to a temporary file
#         with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(file.file.read())
#             temp_file_path = temp_file.name

#         # Load the PDF
#         loader = PyMuPDFLoader(temp_file_path)
#         docs = loader.load()

#         # Process the document
#         document_text = " ".join(doc.page_content for doc in docs)
#         chunks = split_by_headings(document_text, headers)

#         # Convert chunks into Document objects
#         splits = [
#             Document(page_content=chunk["content"],
#                      metadata={"heading": chunk["heading"]})
#             for chunk in chunks
#         ]

#         # Create the vectorstore
#         vectorstore = Chroma.from_documents(documents=splits,
#                                             embedding=OpenAIEmbeddings())
#         return {"message": "PDF uploaded and processed successfully."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    try:
        # Clear any previous session data (vectorstore)
        vectorstore = None  # Reset the vectorstore to clear the previous document's embeddings

        # Save the uploaded file to a temporary file
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Load the PDF
        loader = PyMuPDFLoader(temp_file_path)
        docs = loader.load()

        # Process the document
        document_text = " ".join(doc.page_content for doc in docs)
        chunks = split_by_headings(document_text, headers)

        # Convert chunks into Document objects
        splits = [
            Document(page_content=chunk["content"],
                     metadata={"heading": chunk["heading"]})
            for chunk in chunks
        ]

        # Create the vectorstore (reinitialize it)
        vectorstore = Chroma.from_documents(documents=splits,
                                            embedding=OpenAIEmbeddings())

        # Optionally, delete the temporary file after processing
        os.remove(temp_file_path)

        return {"message": "PDF uploaded and processed successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
also when you are asked what you are trained on or which document do you contain say you were trained on the prudential guidelines, Access bank's CPG and the Basel Frameworks, note that this quesion can come in different ways \
If you don't know the answer or if the answer is not in the document you are trained on, just say that the answer is not in the content provided and do not answer the question . \
ALWAYS provide the reference document(s) for your answer.

just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
qa_system_prompt = """You are a financial analysis assistant specializing in Facility Approval Memos (FAMs). \
Your task is to provide a detailed and comprehensive evaluation of the FAM based on the provided context. \
Your responses must be thorough, precise, and grounded in the information from the FAM. If the required information is not present in the context, explicitly state that it is missing and suggest what additional information would be needed.

For every question or request:
- Provide a detailed explanation that addresses all aspects of the question.
- Reference the relevant section(s) of the FAM to support your response.
- Highlight any gaps, inconsistencies, or missing details in the provided context.
- Offer actionable recommendations or insights where applicable.

Ensure your analysis is professional, well-structured, and adheres to financial analysis standards.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


@app.post("/ask")
async def ask_question(question: Question):
    global vectorstore

    if vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded. Please upload a PDF first.")

    try:
        retriever = vectorstore.as_retriever()
        history_aware_retriever = create_history_aware_retriever(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0), retriever,
            contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0), qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,
                                           question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        result = conversational_rag_chain.invoke(
            {"input": question.query},
            config={"configurable": {
                "session_id": question.session_id
            }},
        )

        get_session_history(question.session_id).add_user_message(
            question.query)
        get_session_history(question.session_id).add_ai_message(
            result["answer"])

        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)