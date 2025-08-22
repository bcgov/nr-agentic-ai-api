from langchain.tools import tool

@tool("echo_tool")
def echo_tool(query: str) -> str:
    """Echoes the input query."""
    return query
