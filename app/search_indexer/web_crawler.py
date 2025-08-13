from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
)  # Add more loaders as needed
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import glob
from azure.storage.blob import BlobServiceClient
from langchain_community.document_loaders import AzureBlobStorageFileLoader
import traceback


# Environment variables (will be set in Azure Functions later)
AZURE_STORAGE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={os.getenv('AZURE_STORAGE_ACCOUNT_NAME')};AccountKey={os.getenv('AZURE_STORAGE_ACCOUNT_KEY')};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)
container_client = blob_service_client.get_container_client(
    os.getenv("AZURE_STORAGE_CONTAINER_NAME")
)
# Load environment variables (use dotenv if preferred)
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",  # Deploy this embedding model in Azure OpenAI if not already (similar to GPT deployment)
    openai_api_version="2024-02-01",  # Adjust to latest
)
vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
    index_name="bc-water-index",  # Create if doesn't exist
    embedding_function=embeddings.embed_query,
    search_type="hybrid",  # Enables vector + keyword
)

def start_indexing():
    try:
        # Load and chunk documents
        docs = []
        print("Starting document loading from blob storage...")

        for blob in container_client.list_blobs():
            try:
                if blob.name.endswith(".pdf"):
                    print(f"Loading PDF: {blob.name}")
                    loader = AzureBlobStorageFileLoader(
                        conn_str=AZURE_STORAGE_CONNECTION_STRING,
                        container=os.getenv("AZURE_STORAGE_CONTAINER_NAME"),
                        blob_name=blob.name,
                    )
                    docs.extend(loader.load())
                    print(f"Successfully loaded PDF: {blob.name}")
                elif blob.name.endswith(".html") or blob.name.endswith(".txt"):
                    print(f"Loading file: {blob.name}")
                    # Use the same loader or WebBaseLoader if it's a URL, but since it's blob, use file loader
                    loader = AzureBlobStorageFileLoader(
                        conn_str=AZURE_STORAGE_CONNECTION_STRING,
                        container=os.getenv("AZURE_STORAGE_CONTAINER_NAME"),
                        blob_name=blob.name,
                    )
                    docs.extend(loader.load())
                    print(f"Successfully loaded file: {blob.name}")
            except Exception as blob_error:
                print(f"Error loading blob {blob.name}: {str(blob_error)}")
                print(f"Blob error type: {type(blob_error).__name__}")
                continue

        # Example for webpage
        try:
            print("Loading webpage...")
            web_loader = WebBaseLoader(
                "https://portalext.nrs.gov.bc.ca/web/client/-/unit-converter"
            )
            docs.extend(web_loader.load())
            print("Successfully loaded webpage")
        except Exception as web_error:
            print(f"Error loading webpage: {str(web_error)}")
            print(f"Webpage error type: {type(web_error).__name__}")

        print(f"Total documents loaded: {len(docs)}")

        # Add more loaders for other files...
        print("Starting text splitting...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks from documents")

        # Index
        print("Starting vector store indexing...")
        vector_store.add_documents(chunks)
        print("Successfully indexed all chunks")

        return {"message": "Indexing completed successfully"}

    except Exception as e:
        print(f"Critical error during indexing: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Full traceback: {traceback.format_exc()}")
        return {"error": f"Indexing failed: {str(e)}"}
