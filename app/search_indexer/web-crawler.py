from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
)  # Add more loaders as needed
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob

# Load environment variables (use dotenv if preferred)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",  # Deploy this embedding model in Azure OpenAI if not already (similar to GPT deployment)
    openai_api_version="2023-05-15",  # Adjust to latest
)
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
    index_name="bc-water-index",  # Create if doesn't exist
    embedding_function=embeddings.embed_query,
    search_type="hybrid",  # Enables vector + keyword
)


def start_indexing():
    # Load and chunk documents
    docs = []
    # Example for PDF
    # Load all documents from source_docs folder
    for file_path in glob.glob("./source_docs/*"):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        # Add other file types as needed
        # elif file_path.endswith('.txt'):
        #     loader = TextLoader(file_path)
        #     docs.extend(loader.load())
    # Example for webpage
    web_loader = WebBaseLoader(
        "https://portalext.nrs.gov.bc.ca/web/client/-/unit-converter"
    )
    docs.extend(web_loader.load())
    # Add more loaders for other files...
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    # Index
    vector_store.add_documents(chunks)
    return {"message": "Indexing completed"}
