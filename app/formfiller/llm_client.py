"""
Azure OpenAI LLM client and initialization logic for agentic flows.
"""
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

# Global variable to hold the LLM instance
_llm: Optional[AzureChatOpenAI] = None

def get_llm() -> AzureChatOpenAI:
    """
    Get or initialize the Azure OpenAI LLM client.
    This lazy initialization prevents import-time failures if credentials aren't available.
    """
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            openai_api_version="2024-12-01-preview",
        )
    return _llm

# For backward compatibility, create a module-level property that returns the actual LLM when accessed
class ModuleLLM:
    def __getattr__(self, name):
        # When any attribute is accessed, return the real LLM instance
        return getattr(get_llm(), name)
    
    def __call__(self, *args, **kwargs):
        # When called directly, call the real LLM
        return get_llm()(*args, **kwargs)
    
    def __class__(self):
        # Return the class of the real LLM for isinstance checks
        return get_llm().__class__
    
    def __bool__(self):
        # Always return True so it can be used in boolean contexts
        return True

# Create a module-level llm that behaves like the real LLM but initializes lazily
llm = ModuleLLM()

