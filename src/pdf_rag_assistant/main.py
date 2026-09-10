from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF RAG Assistant")
st.caption("Upload a PDF and ask questions about its content.")


load_dotenv()


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf"
)

if uploaded_file is None:
    st.stop()

file_signature = (uploaded_file.name, uploaded_file.size)
if st.session_state.get("current_file") != file_signature:
    st.session_state.current_file = file_signature
    st.session_state.query = ""

try:
    reader = PdfReader(uploaded_file)
    pages = reader.pages
except Exception:
    st.error("Could not read this PDF. The file may be corrupted or password-protected.")
    st.stop()

if len(pages)> 40:
    st.error("Please upload a PDF with 40 pages or fewer.")
    st.stop()

st.caption(f"📄 {uploaded_file.name} · {len(pages)} pages")

# Extract text from the PDF
pdf_text = ""

for page in pages:
    text = page.extract_text()

    if text:
        pdf_text += text + "\n"

if not pdf_text.strip():
    st.error("No readable text was found in this PDF.")
    st.stop()

# Split the text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(pdf_text)


# Create embeddings and build the FAISS index
@st.cache_resource(show_spinner=False, max_entries=1)
def create_vector_store(text_chunks):

    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    faiss_index = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return faiss_index


with st.spinner("Processing your PDF..."):
    vector_store = create_vector_store(chunks)


query = st.text_input("Ask a question about the PDF", key="query")
if not query:
    st.stop()

with st.spinner("Searching the PDF and generating an answer..."):
    # Retrieve the 5 most relevant chunks
    results = vector_store.similarity_search(
        query,
        k=5
    )

    context = "\n\n".join(
        result.page_content for result in results
    )

    prompt = f"""
    Answer the question using only the context provided below.
    If the answer cannot be found in the context, say:
    "I could not find this information in the PDF" 
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """


    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )

    try:
        response = llm.invoke(prompt)
    except Exception:
        st.error("Could not generate an answer. Please try again later.")
        st.stop()


st.subheader("Answer")

if isinstance(response.content, list):
    st.write(response.content[0]["text"])
else:
    st.write(response.content)


with st.expander("🔎 View retrieved context"):
    for i, result in enumerate(results, 1):
        st.markdown(f"**Chunk {i}**")
        st.write(result.page_content)
