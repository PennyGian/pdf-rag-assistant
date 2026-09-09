# PDF RAG Assistant

A simple Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content.
**Live Demo:** [Try the app on Streamlit](https://pdf-rag-assistant-lvkgeastzlerw6ks2sappjf.streamlit.app/)

The application extracts the text from the PDF, splits it into smaller chunks, creates embeddings, and indexes them with FAISS. When the user asks a question, the most relevant chunks are retrieved and provided as context to a Gemini model to generate an answer.

## Features

- Upload and process text-based PDF files
- Split PDF text into overlapping chunks
- Generate embeddings using FastEmbed
- Index and search embeddings with FAISS
- Retrieve the 5 most relevant chunks for each question
- Generate answers using Gemini based only on the retrieved context
- View the retrieved chunks used to generate the answer

## Tech Stack

- Python
- Streamlit
- LangChain
- FastEmbed (`BAAI/bge-small-en-v1.5`)
- FAISS
- Google Gemini
- PyPDF

## How It Works

1. The user uploads a PDF.
2. PyPDF extracts the text from the document.
3. The text is split into overlapping chunks.
4. FastEmbed converts the chunks into embeddings, which are indexed with FAISS.
5. When a question is asked, it is converted into an embedding and FAISS retrieves the 5 most relevant chunks using similarity search.
6. The retrieved chunks are passed to Gemini as context together with the question.
7. Gemini generates an answer based only on the provided context.

## Limitations

- Scanned or image-only PDFs are not supported because the application does not use OCR.
- Answer quality depends on whether the relevant information is included in the retrieved chunks.
- Broad or multi-part questions may require information from more chunks than the current retrieval setup provides.
- The application currently processes one PDF at a time.

## Run Locally

Clone the repository and install the dependencies:

```bash
git clone https://github.com/PennyGian/pdf-rag-assistant.git
cd pdf-rag-assistant
uv sync
```

Create a `.env` file in the project root and add your Google API key:

```text
GOOGLE_API_KEY=your_api_key_here
```

Run the Streamlit application:

```bash
uv run streamlit run src/pdf_rag_assistant/main.py
```

## Project Structure

```text
pdf-rag-assistant/
├── .streamlit/
├── src/
│   └── pdf_rag_assistant/
│       ├── __init__.py
│       └── main.py
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
└── uv.lock
```