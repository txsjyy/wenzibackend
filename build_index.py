from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def build_vectorstore():
    # Load and parse JSON manually
    with open("training_data.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Each entry is assumed to have a "data" field with a string
    docs = []
    for item in raw_data:
        content = item.get("data")
        if isinstance(content, str):  # only include valid string entries
            docs.append(Document(page_content=content))

    # Split content into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # Generate embeddings
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding_model)

    # Save index locally
    vectorstore.save_local("faiss_index")

if __name__ == "__main__":
    build_vectorstore()
