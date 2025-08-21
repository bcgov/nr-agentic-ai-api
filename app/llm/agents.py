"""
Agent definitions and executor setup for LLM-based flows.
"""
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import llm

# Example tool (can be extended)
@tool("echo_tool")
def echo_tool(query: str) -> str:
    """Echoes the input query."""
    return query

# Land agent setup
land_tools = [echo_tool]
land_prompt = PromptTemplate.from_template(
    "You are the Land agent. You can use tools to answer the user's request.\n\n"
    "You have access to the following tools:\n{tools}\n\n"
    "When deciding what to do, follow this format exactly:\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, must be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation cycle can repeat) ...\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "{agent_scratchpad}"
)
land_agent = create_react_agent(llm, land_tools, land_prompt)
land_executor = AgentExecutor(
    agent=land_agent,
    tools=land_tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=4,
)

# Water agent setup (can be extended)
water_tools = [echo_tool]
water_prompt = PromptTemplate.from_template(
    "You are the Water agent. You can use tools to answer the user's request.\n\n"
    "You have access to the following tools:\n{tools}\n\n"
    "When deciding what to do, follow this format exactly:\n"
    "Question: the input question you must answer\n"
    "Thought: you should always think about what to do\n"
    "Action: the action to take, must be one of [{tool_names}]\n"
    "Action Input: the input to the action\n"
    "Observation: the result of the action\n"
    "... (this Thought/Action/Action Input/Observation cycle can repeat) ...\n"
    "Thought: I now know the final answer\n"
    "Final Answer: the final answer to the original input question\n\n"
    "Begin!\n\n"
    "Question: {input}\n"
    "{agent_scratchpad}"
)
water_agent = create_react_agent(llm, water_tools, water_prompt)
water_executor = AgentExecutor(
    agent=water_agent,
    tools=water_tools,
    verbose=False,
    handle_parsing_errors=True,
    max_iterations=4,
)
