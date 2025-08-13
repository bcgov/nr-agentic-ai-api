"""
Web Crawler and Document Indexer using Azure Document Intelligence

This module processes documents from Azure Blob Storage using Azure Document Intelligence
for advanced document parsing and analysis, then indexes them in Azure AI Search.

Required Environment Variables:
- AZURE_STORAGE_ACCOUNT_NAME: Azure Storage Account name
- AZURE_STORAGE_ACCOUNT_KEY: Azure Storage Account access key
- AZURE_STORAGE_CONTAINER_NAME: Container name containing documents
- AZURE_SEARCH_ENDPOINT: Azure AI Search service endpoint
- AZURE_SEARCH_ADMIN_KEY: Azure AI Search admin key
- AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT: Document Intelligence service endpoint
- AZURE_DOCUMENT_INTELLIGENCE_KEY: Document Intelligence service key (optional if using managed identity)
"""

from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, ContentFormat
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
import traceback


# Environment variables (will be set in Azure Functions later)
AZURE_STORAGE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={os.getenv('AZURE_STORAGE_ACCOUNT_NAME')};AccountKey={os.getenv('AZURE_STORAGE_ACCOUNT_KEY')};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)
container_client = blob_service_client.get_container_client(
    os.getenv("AZURE_STORAGE_CONTAINER_NAME")
)

# Initialize Document Intelligence client
document_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
document_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

# Use key-based authentication if available, otherwise use DefaultAzureCredential
if document_intelligence_key:
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=document_intelligence_endpoint,
        credential=AzureKeyCredential(document_intelligence_key),
    )
else:
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=document_intelligence_endpoint, credential=DefaultAzureCredential()
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


def process_document_with_intelligence(blob_name, blob_data):
    """
    Process a document using Azure Document Intelligence
    Returns a Document object with the extracted content
    """
    try:
        # Create the request with the blob data
        analyze_request = AnalyzeDocumentRequest(bytes_source=blob_data)

        # Use the prebuilt-read model for general document reading
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-read",
            analyze_request,
            output_content_format=ContentFormat.MARKDOWN,
        )
        result = poller.result()

        # Extract the content
        content = result.content if result.content else ""

        # Create a Document object
        return Document(
            page_content=content,
            metadata={
                "source": blob_name,
                "processed_with": "azure_document_intelligence",
            },
        )
    except Exception as e:
        print(
            f"Error processing document {blob_name} with Document Intelligence: {str(e)}"
        )
        return None


def start_indexing():
    try:
        # Load and chunk documents
        docs = []
        print("Starting document loading from blob storage...")

        for blob in container_client.list_blobs():
            try:
                # Get the blob data
                blob_client = container_client.get_blob_client(blob.name)
                blob_data = blob_client.download_blob().readall()

                if blob.name.endswith((".pdf", ".docx", ".doc", ".txt", ".html")):
                    print(f"Processing document: {blob.name}")

                    # Use Document Intelligence to process the document
                    document = process_document_with_intelligence(blob.name, blob_data)

                    if document:
                        docs.append(document)
                        print(f"Successfully processed document: {blob.name}")
                    else:
                        print(f"Failed to process document: {blob.name}")

                else:
                    print(f"Skipping unsupported file type: {blob.name}")

            except Exception as blob_error:
                print(f"Error processing blob {blob.name}: {str(blob_error)}")
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
