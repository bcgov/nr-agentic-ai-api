"""
Water agent definition and executor setup.
"""
from langchain.agents import create_react_agent, AgentExecutor
from ..llm_client import llm
from app.llm.tools.echo_tool import echo_tool
from app.llm.prompts.water_prompt import water_prompt

water_tools = [echo_tool]
water_agent = create_react_agent(llm, water_tools, water_prompt)
water_executor = AgentExecutor(
    agent=water_agent,
    tools=water_tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=4,
)
