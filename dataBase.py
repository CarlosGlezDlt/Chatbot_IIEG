import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
import shutil


hf_token = os.environ["HF_TOKEN"]


def load_documents():
    loader = DirectoryLoader(
        path='Documents',
        glob='**/*.pdf',  # Match all PDF files in subdirectories
        loader_cls=PyPDFLoader  # Use the PyPDFLoader class for PDF files
    )
    return loader.load()


def splitter_function(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def createDb(db: str, collection_name: str):
    docs = load_documents()

    chunks = splitter_function(docs)

    chunks_id = calculate_chunk_ids(chunks)

    vector_store = Chroma.from_documents(
        chunks_id,
        embedding = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
        persist_directory=db,
        collection_name=collection_name
    )
    return vector_store


def clearDb(path):
    if os.path.exists(path):
        shutil.rmtree(path)


clearDb('Db')


Vector_store = createDb('Db', 'Documents')