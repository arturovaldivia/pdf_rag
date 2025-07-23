
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#----- Importing libraries ----------------------------------------------------------------------------------
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser

import os
import tempfile
import uuid
import pandas as pd
import re
import requests

#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#----- Utility Functions -----------------------------------------------------------------------------------
#////////////////////////////////////////////////////////////////////////////////////////////////////////////

def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename


def get_pdf_text(uploaded_file): 
    """
    Load a PDF document from an uploaded file and return it as a list of documents

    Parameters:
        uploaded_file (file-like object): The uploaded PDF file to load

    Returns:
        list: A list of documents created from the uploaded PDF file
    """
    try:
        # Read file content
        input_file = uploaded_file.read()

        # Create a temporary file (PyPDFLoader requires a file path to read the PDF,
        # it can't work directly with file-like objects or byte streams that we get from Streamlit's uploaded_file)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(input_file)
        temp_file.close()

        # load PDF document
        loader = PyPDFLoader(temp_file.name)
        documents = loader.load()

        return documents
    
    finally:
        # Ensure the temporary file is deleted when we're done with it
        os.unlink(temp_file.name)


#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#----- Embedding and Vector Store Functions -----------------------------------------------------------------
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
class SentenceTransformerEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name
                                         , token=False
                                         )  # ✅ Use only short name

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def split_document(documents, chunk_size, chunk_overlap):    
    """
    Function to split generic text into smaller chunks.
    chunk_size: The desired maximum size of each chunk (default: 400)
    chunk_overlap: The number of characters to overlap between consecutive chunks (default: 20).

    Returns:
        list: A list of smaller text chunks created from the generic text
    """
    text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                length_function=len,
                                separators=["\n\n", "\n", " "]
                                )

    #return text_splitter.split_documents(documents)
    split_docs = text_splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_index"] = i
        
    return split_docs


def get_embedding_function(api_key=None):  # Keep api_key for compatibility
    return SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="db"):

    """
    Create a vector store from a list of text chunks.

    :param chunks: A list of generic text chunks
    :param embedding_function: A function that takes a string and returns a vector
    :param file_name: The name of the file to associate with the vector store
    :param vector_store_path: The directory to store the vector store

    :return: A Chroma vector store object
    """

    # Create a list of unique ids for each document based on the content
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    # Ensure that only unique docs with unique ids are kept
    unique_ids = set()
    unique_chunks = []
    
    unique_chunks = [] 
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk)        

    # Create a new Chroma database from the documents
    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        collection_name=clean_filename(file_name),
                                        embedding=embedding_function, 
                                        ids=list(unique_ids), 
                                        persist_directory = vector_store_path)

    # The database should save automatically after we create it
    # but we can also force it to save using the persist() method
    vectorstore.persist()
    
    return vectorstore


def create_vectorstore_from_texts(documents, api_key, file_name):
    
    # Step 2 split the documents  
    """
    Create a vector store from a list of texts.

    :param documents: A list of generic text documents
    :param api_key: The OpenAI API key used to create the vector store
    :param file_name: The name of the file to associate with the vector store

    :return: A Chroma vector store object
    """
    docs = split_document(documents, chunk_size=1000, chunk_overlap=200)
    print(f"\n\nNumber of chunks created: {len(docs)}\n\n")
    # Step 3 define embedding function
    embedding_function = get_embedding_function(api_key)
    print(f"\n\n-----Embedding function created------\n\n")
    # Step 4 create a vector store  
    vectorstore = create_vectorstore(docs, embedding_function, file_name)
    
    return vectorstore


def load_vectorstore(file_name, api_key, vectorstore_path="db"):

    """
    Load a previously saved Chroma vector store from disk.

    :param file_name: The name of the file to load (without the path)
    :param api_key: The OpenAI API key used to create the vector store
    :param vectorstore_path: The path to the directory where the vector store was saved (default: "db")
    
    :return: A Chroma vector store object
    """
    embedding_function = get_embedding_function(api_key)
    return Chroma(persist_directory=vectorstore_path, 
                  embedding_function=embedding_function, 
                  collection_name=clean_filename(file_name))

#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#----- Query Document Functions -----------------------------------------------------------------------------
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
# Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
Respond ONLY with valid JSON in this format:
{{
  "paper_title": {{"answer": "...", "sources": "...", "reasoning": "..."}},
  "paper_summary": {{"answer": "...", "sources": "...", "reasoning": "..."}},
  "publication_year": {{"answer": "...", "sources": "...", "reasoning": "..."}},
  "paper_authors": {{"answer": "...", "sources": "...", "reasoning": "..."}}
}}
"""

class AnswerWithSources(BaseModel):
    """An answer to the question, with sources and reasoning."""
    answer: str = Field(description="Answer to question")
    sources: str = Field(description="Full direct text chunk from the context used to answer the question")
    reasoning: str = Field(description="Explain the reasoning of the answer based on the sources")
    
class ExtractedInfoWithSources(BaseModel):
    """Extracted information about the research article"""
    paper_title: AnswerWithSources
    paper_summary: AnswerWithSources
    publication_year: AnswerWithSources
    paper_authors: AnswerWithSources

def format_docs(docs):
    """
    Format a list of Document objects into a single string.

    :param docs: A list of Document objects

    :return: A string containing the text of all the documents joined by two newlines
    """
    return "\n\n".join(doc.page_content for doc in docs)

# retriever | format_docs passes the question through the retriever, generating Document objects, and then to format_docs to generate strings;
# RunnablePassthrough() passes through the input question unchanged.

def query_document(vectorstore, query, api_key=None):
    """
    Query a vector store with a question and return a structured response using Ollama.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store

    :return: A pandas DataFrame with three rows: 'answer', 'source', and 'reasoning'
    """

    llm = ChatOllama(model="llama3.1")  # Local model

    retriever = vectorstore.as_retriever(search_type="similarity"
                                         , search_kwargs={
                                                        'filter': {'page_label':{"$in": ["1"]}}
                                                        }
                                         )

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    parser = PydanticOutputParser(pydantic_object=ExtractedInfoWithSources)

    retrieved_docs = retriever.get_relevant_documents(query)

    # Print to inspect
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Document {i} ---")
        print(doc.page_content[:500])  # preview first 500 characters
        print("Metadata:", doc.metadata)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | parser
    )

    structured_response = rag_chain.invoke(query)

    df = pd.DataFrame([structured_response.dict()])

    # Reorganize into a DataFrame with answer/source/reasoning per field
    answer_row, source_row, reasoning_row = [], [], []

    for col in df.columns:
        answer_row.append(df[col][0]['answer'])
        source_row.append(df[col][0]['sources'])
        reasoning_row.append(df[col][0]['reasoning'])

    structured_response_df = pd.DataFrame(
        [answer_row, source_row, reasoning_row],
        columns=df.columns,
        index=['answer', 'source', 'reasoning']
    )

    return structured_response_df.T, retriever

def query_document_improved(vectorstore, query, api_key=None):
    """
    Query a vector store with a question and return a structured response using Ollama.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store

    :return: A pandas DataFrame with three rows: 'answer', 'source', and 'reasoning'
    """

    llm = ChatOllama(model="llama3.1")  # Local model

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                         search_kwargs={"score_threshold": 0.5
                                                        , 'filter': {'page_label':1}
                                                        }
                                        )

    retrieved_docs = retriever.get_relevant_documents(query)

    # Print to inspect
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Document {i} ---")
        print(doc.page_content)  # preview first 500 characters
        print("Metadata:", doc.metadata)

    return retrieved_docs


#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#----- Get the response from local model via Ollama ---------------------------------------------------------
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
def get_ollama_response(prompt, server_url = "http://localhost", port=11434, model = "deepseek-llm:7b"):
  """
  Sends a prompt to the Ollama Llama 3 model and returns the response.

  Args:
    prompt: The input prompt.

  Returns:
    The generated text from local model (e.g., llama3.1) via Ollama.

  How to use:
    prompt = "What's colder than ice?"
    response = get_ollama_response(prompt)
    print(response)
  """

  url = f"{server_url}:{port}/api/chat"
  headers = {"Content-Type": "application/json"}

  data = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "stream": False
  }

  response = requests.post(url, headers=headers, json=data)

  response_json = response.json()

  return response_json["message"]["content"]


#////////////////////////////////////////////////////////////////////////////////////////////////////////////
#----- Get the response from OpenAI model -------------------------------------------------------------------
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
def get_embedding_function_openai(api_key):
    """
    Return an OpenAIEmbeddings object, which is used to create vector embeddings from text.
    The embeddings model used is "text-embedding-ada-002" and the OpenAI API key is provided
    as an argument to the function.

    Parameters:
        api_key (str): The OpenAI API key to use when calling the OpenAI Embeddings API.

    Returns:
        OpenAIEmbeddings: An OpenAIEmbeddings object, which can be used to create vector embeddings from text.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=api_key
    )
    return embeddings

def query_document_openai(vectorstore, query, api_key):

    """
    Query a vector store with a question and return a structured response.

    :param vectorstore: A Chroma vector store object
    :param query: The question to ask the vector store
    :param api_key: The OpenAI API key to use when calling the OpenAI Embeddings API

    :return: A pandas DataFrame with three rows: 'answer', 'source', and 'reasoning'
    """
    print("\n\n We're in the query_document function \n\n")
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    retriever=vectorstore.as_retriever(search_type="similarity")

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm.with_structured_output(ExtractedInfoWithSources, strict=True)
        )

    structured_response = rag_chain.invoke(query)
    df = pd.DataFrame([structured_response.dict()])

    # Transforming into a table with two rows: 'answer' and 'source'
    answer_row = []
    source_row = []
    reasoning_row = []

    for col in df.columns:
        answer_row.append(df[col][0]['answer'])
        source_row.append(df[col][0]['sources'])
        reasoning_row.append(df[col][0]['reasoning'])

    # Create new dataframe with two rows: 'answer' and 'source'
    structured_response_df = pd.DataFrame([answer_row, source_row, reasoning_row], columns=df.columns, index=['answer', 'source', 'reasoning'])
  
    return structured_response_df.T
