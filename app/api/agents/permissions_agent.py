import os
from typing import Optional, List
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from app.core.logging import get_logger  # Assuming shared logging module
from app.core.config import (
    settings,
)  # Assuming mapping document is loaded from a config module

MAPPING_DOC = settings.MAPPING_DOC  # Load mapping document from settings
logger = get_logger(__name__)
load_dotenv()

# Initialize the Azure OpenAI LLM (shared with Orchestrator)
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version="2024-12-01-preview",
)


# Define the AI Search Tool (shared tool, already supports async via _arun)
class AISearchTool(BaseTool):
    """Tool for searching and retrieving data from AI Search index."""

    name: str = "ai_search_tool"
    description: str = "A tool that searches and retrieves data from AI Search index"
    search_client: Optional[SearchClient] = None

    def __init__(self):
        super().__init__()
        # Get environment variables
        search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        search_key = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
        index_name = "bc-water-index"  # hardcoded, consistent with Orchestrator
        # Initialize search client if environment variables are set
        if search_endpoint and search_key and index_name:
            credential = AzureKeyCredential(search_key)
            self.search_client = SearchClient(
                endpoint=search_endpoint, index_name=index_name, credential=credential
            )

    def _run(self, query: str, run_manager=None) -> str:
        if not self.search_client:
            return "Azure Search not configured."
        try:
            # Search the index
            search_results = self.search_client.search(
                search_text=query, select=["*"], top=5
            )
            # Process results
            results = []
            for result in search_results:
                results.append(dict(result))
            if not results:
                return f"No results found for query '{query}'"
            return str(results)
        except Exception as e:
            return f"Error searching index: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


class PermissionsCheckTool(BaseTool):
    """Custom tool for checking permissions and compliance, including fee exemptions based on mapping doc."""

    name: str = "permissions_check_tool"
    description: str = "A tool that checks permissions, compliance, and fee exemption eligibility using the provided mapping document and user input."

    def _run(self, query: str, run_manager=None) -> str:
        try:
            # Simple logic: Check if query relates to fee exemption and validate against mapping
            # In production, expand this to more sophisticated validation/rules engine
            if "fee exemption" in query.lower():
                required_fields = [
                    field["domElementId"]
                    for field in MAPPING_DOC["ApplicantInformation"]
                    if field["required"]
                ]
                return f"Fee exemption check: Required fields are {required_fields}. Validate user input against these. Mapping doc: {MAPPING_DOC}"
            else:
                return "No specific permissions check matched. Use general search if needed."
        except Exception as e:
            return f"Error in permissions check: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)


# Form field model (matched to Orchestrator for compatibility)
class FormField(BaseModel):
    """Model for individual form fields"""

    data_id: str = None
    fieldLabel: str = None
    fieldType: str = None
    fieldValue: str = None


# Create the Permissions agent (manages access, compliance, and fee exemptions using mapping doc)
permissions_tools = [AISearchTool(), PermissionsCheckTool()]
permissions_prompt = PromptTemplate.from_template(
    "You are a Permissions agent. Your role is to manage access, compliance, and check eligibility (e.g., fee exemptions) "
    "using the mapping document and available tools. "
    "If form fields are provided, validate them against the mapping doc (e.g., check required fields for exemptions).\n\n"
    "User request: {input}\n\n"
    "Available tools: {tools}\n\nTool names: {tool_names}\n\n{agent_scratchpad}"
)
permissions_agent = create_react_agent(llm, permissions_tools, permissions_prompt)
permissions_agent_executor = AgentExecutor(
    agent=permissions_agent, tools=permissions_tools, verbose=True
)


# Async function to invoke the Permissions Agent (called by Orchestrator's workflow node)
async def invoke_permissions_agent(
    input_str: str, form_fields: Optional[List[FormField]] = None
) -> str:
    """Async invocation function for the Permissions Agent, compatible with Orchestrator's delegation.
    Handles optional form_fields by appending them to the input for processing.
    Returns a string output suitable for frontend chat (e.g., human-readable response).
    """
    logger.info(f"Invoking Permissions Agent with input: {input_str}")

    # Process form fields if provided (e.g., serialize and append to input)
    enhanced_input = input_str
    if form_fields:
        form_data_str = "\nForm fields provided:\n" + "\n".join(
            [
                f"- {field.fieldLabel or field.data_id}: {field.fieldValue}"
                for field in form_fields
                if field.fieldValue
            ]
        )
        enhanced_input += form_data_str
        logger.info(f"Enhanced input with form fields: {enhanced_input}")

    try:
        result = await permissions_agent_executor.ainvoke({"input": enhanced_input})
        output = result["output"]
        # Format output for frontend chat (e.g., prefix with a label for clarity)
        formatted_output = f"Permissions Agent Response: {output}"
        logger.info("Permissions Agent execution successful")
        return formatted_output
    except Exception as e:
        logger.error(f"Error in Permissions Agent: {str(e)}")
        return f"Error in Permissions Agent: {str(e)} (Please check logs for details)"
