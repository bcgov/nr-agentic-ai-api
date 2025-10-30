from ..llm_client import get_llm
from app.formfiller.prompts.process_field_prompt import process_field_prompt
from langchain.chains import LLMChain

# You don't need tools or ReAct agent
process_field_executor = LLMChain(
    llm=get_llm(),
    prompt=process_field_prompt,
    verbose=True
)