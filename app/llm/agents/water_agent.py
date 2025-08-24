"""
Water agent definition and executor setup.
"""
from langchain.agents import create_react_agent, AgentExecutor
from ..llm_client import llm
from app.llm.tools.echo_tool import echo_tool
from app.llm.tools.form_verification_tool import verify_fee_exemption_form
from app.llm.prompts.water_prompt import water_prompt
from app.llm.verification_agent_llm import VerificationAgentLLM

water_tools = [echo_tool, verify_fee_exemption_form]
water_agent = create_react_agent(llm, water_tools, water_prompt)
water_executor = AgentExecutor(
    agent=water_agent,
    tools=water_tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
    return_intermediate_steps=True,
)

# Add verification to water agent workflow
def verify_water_form(form_fields):
    verifier = VerificationAgentLLM()
    return verifier.verify_form(form_fields)
