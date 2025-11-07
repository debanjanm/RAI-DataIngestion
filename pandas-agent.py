# Installation
# pip install langchain-cohere langchain-core pandas

from langchain_cohere import ChatCohere
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
import pandas as pd
import cohere

# Load CSV
df = pd.read_csv("master_metadata.csv")

# Define custom tools
def query_dataframe(query: str) -> str:
    """Execute pandas query on dataframe"""
    try:
        result = df.query(query)
        return result.to_string()
    except Exception as e:
        return f"Error: {str(e)}"

def describe_dataframe(column: str = None) -> str:
    """Get statistical description of dataframe or specific column"""
    if column:
        return df[column].describe().to_string()
    return df.describe().to_string()

def get_column_info(dummy: str = "") -> str:
    """Get information about dataframe columns"""
    info = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "shape": df.shape
    }
    return str(info)

# Create tools
tools = [
    Tool(
        name="QueryDataFrame",
        func=query_dataframe,
        description="Execute pandas query operations on the CSV data. Use pandas query syntax."
    ),
    Tool(
        name="DescribeDataFrame",
        func=describe_dataframe,
        description="Get statistical summary of the dataframe or a specific column."
    ),
    Tool(
        name="GetColumnInfo",
        func=get_column_info,
        description="Get information about dataframe structure, columns, and data types."
    )
]

# Initialize Cohere LLM
llm = ChatCohere(
    model="command-r-plus",
    temperature=0,
    cohere_api_key="your_cohere_api_key"
)

# Create ReAct agent with custom prompt
react_prompt = PromptTemplate.from_template("""
Answer the following question using the available tools.

You have access to these tools:
{tools}

Use this format:
Question: the input question
Thought: consider what to do
Action: the action to take (must be one of {tool_names})
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}
Thought: {agent_scratchpad}
""")

# Create agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=15
)

# Execute query
result = agent_executor.invoke({
    "input": "What are the column names and their data types?"
})
print(result["output"])
