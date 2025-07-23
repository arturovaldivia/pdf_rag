import streamlit as st  
import os
from functions import *
import base64

# Initialize the API key in session state if it doesn't exist
if 'huggingface_token' not in st.session_state:
    st.session_state.huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')

def display_pdf(uploaded_file):

    """
    Display a PDF file that has been uploaded to Streamlit.

    The PDF will be displayed in an iframe, with the width and height set to 700x1000 pixels.

    Parameters
    ----------
    uploaded_file : UploadedFile
        The uploaded PDF file to display.

    Returns
    -------
    None
    """
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()
    
    # Convert to Base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    
    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    
    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)


def load_streamlit_page():

    """
    Load the streamlit page with two columns. The left column contains a text input box for the user to input their OpenAI API key, and a file uploader for the user to upload a PDF document. The right column contains a header and text that greet the user and explain the purpose of the tool.

    Returns:
        col1: The left column Streamlit object.
        col2: The right column Streamlit object.
        uploaded_file: The uploaded PDF file.
    """
    st.set_page_config(layout="wide", page_title="LLM Tool")

    # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
    col1, col2 = st.columns([0.5, 0.5], gap="large")

    with col1:
        st.header("Input your OpenAI API key")
        st.text_input('OpenAI API key', type='password', key='api_key',
                    label_visibility="collapsed", disabled=False)
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type= "pdf")

    return col1, col2, uploaded_file

# Add two tabs to the page
tab1, tab2 = st.tabs(["LLM Tool", "RAG Tool"])

with tab1:
    st.title("LLM Tool")
    # Make a streamlit page
    col1, col2, uploaded_file = load_streamlit_page()

    # Process the input
    if uploaded_file is not None:
        with col2:
            display_pdf(uploaded_file)
            
        # Load in the documents
        documents = get_pdf_text(uploaded_file)
        st.success("Input loaded successfully")

        st.session_state.vector_store = create_vectorstore_from_texts(documents, 
                                                                    api_key=None,
                                                                    file_name=uploaded_file.name
                                                                    )
        st.success("Input Processed")

    # Generate answer
    with col1:
        if st.button("Generate table"):
            with st.spinner("Generating answer"):
                # Load vectorstore:

                answer, aux = query_document(vectorstore = st.session_state.vector_store, 
                                        query = "Give me the title, summary, publication date, and authors of the research paper.",
                                        api_key = None)
                                
                placeholder = st.empty()
                st.write(answer)
                st.warning(format_docs(aux))
        
        #insert a text input box to ask a question
        question = st.text_input("Ask a question about the document:"
                                 , placeholder="What is the main topic of the document?"
                                 , key="question_input"
                                 )
        
        if st.button("Ask question"):
            with st.spinner("Generating answer"):
                if question:
                    #llm = ChatOllama(model="llama3.1")  # Local model

                    all_metas = st.session_state.vector_store._collection.get(include=["metadatas"])["metadatas"]
                    pages = sorted(set(meta["page"] for meta in all_metas if "page" in meta))
                    st.success(pages)

                    retriever = st.session_state.vector_store.as_retriever(search_type="similarity"
                                                                           , search_kwargs={
                                                                                   'filter': {'page_label':{"$in": ["1"]}}
                                                                                   }
                                                                           )

                    retrieved_docs = retriever.get_relevant_documents(question)

                    # Print to inspect
                    for i, doc in enumerate(retrieved_docs):
                        st.write(f"\n--- Document {i} ---")
                        st.write(doc.page_content)  # preview first 500 characters
                        st.write("Metadata:", doc.metadata)

                    #st.write(retrieved_docs)

                    results = st.session_state.vector_store._collection.get(
                                    where={"page_label": "1"},
                                    include=["documents", "metadatas"]
                                )
                    
                    docs = results["documents"]
                    metas = results["metadatas"]

                    df = pd.DataFrame({
                        "chunk": docs,
                        "page": [meta.get("page") for meta in metas],
                        "chunk_index": [meta.get("chunk_index", 0) for meta in metas]
                    })
                    df_sorted = df.sort_values(by="chunk_index").reset_index(drop=True)
                    st.dataframe(df_sorted)  # Or print(df_sorted)



                else:
                    st.error("Please enter a question.")

with tab2:
    st.title("RAG Tool")
    if False:
        # Get the top N relevant chunks
        docs = retriever.get_relevant_documents(query)
        n = 5  # or make this adjustable with a Streamlit slider

        # Convert chunks to a DataFrame
        chunk_data = []
        for i, doc in enumerate(docs[:n]):
            chunk_data.append({
                "Chunk #": i + 1,
                "Page": doc.metadata.get("page", "N/A"),
                "Source": doc.metadata.get("source", "N/A"),
                "Content": doc.page_content[:1000]  # truncate for readability
            })

        df_chunks = pd.DataFrame(chunk_data)

        # Display in Streamlit
        st.subheader("🔍 Retrieved Chunks for Manual Inspection")
        st.dataframe(df_chunks, use_container_width=True)
