"""
Custom Langchain DataFrame Agent using create_agent method
This implementation creates a dataframe agent from scratch without using 
create_pandas_dataframe_agent
"""
import os
import dotenv
dotenv.load_dotenv()


LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT="pandas-agent-project"

from langsmith import traceable

import pandas as pd
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import io
import contextlib

# Sample DataFrame for demonstration
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 75000, 55000, 70000],
    'department': ['Sales', 'Engineering', 'Engineering', 'Sales', 'Marketing']
})

# # Tool 1: Execute Python code on the dataframe
# def python_repl_tool(code: str) -> str:
#     """
#     Execute Python code with access to the dataframe 'df'.
#     Returns the output or error message.
#     """
#     try:
#         # Capture stdout
#         output_buffer = io.StringIO()
        
#         # Create a restricted namespace with the dataframe
#         namespace = {
#             'df': df,
#             'pd': pd,
#             '__builtins__': __builtins__
#         }
        
#         with contextlib.redirect_stdout(output_buffer):
#             # Execute the code
#             exec(code, namespace)
#             result = output_buffer.getvalue()
            
#             # If no output was printed, try to evaluate the last expression
#             if not result:
#                 try:
#                     result = str(eval(code, namespace))
#                 except:
#                     result = "Code executed successfully but produced no output"
        
#         return result if result else "Code executed successfully"
    
#     except Exception as e:
#         return f"Error executing code: {str(e)}"


from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()

# Tool 2: Get dataframe info
def get_df_info(query: str = "") -> str:
    """
    Get information about the dataframe structure, columns, and sample data.
    """
    info_str = f"""
DataFrame Information:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data Types:
{df.dtypes.to_string()}

First 5 rows:
{df.head().to_string()}

Summary Statistics:
{df.describe().to_string()}
"""
    return info_str

# Tool 3: Query specific columns
def query_columns(column_names: str) -> str:
    """
    Query specific columns from the dataframe.
    Input should be comma-separated column names.
    """
    try:
        cols = [col.strip() for col in column_names.split(',')]
        result = df[cols].to_string()
        return result
    except Exception as e:
        return f"Error querying columns: {str(e)}"

# Create tools list
tools = [
    Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    ),
    Tool(
        name="get_dataframe_info",
        func=get_df_info,
        description="""
        Use this tool to get information about the dataframe structure.
        Returns: shape, columns, data types, sample rows, and summary statistics.
        No input required.
        """
    ),
    Tool(
        name="query_columns",
        func=query_columns,
        description="""
        Use this tool to view specific columns from the dataframe.
        Input: comma-separated column names (e.g., "name,age" or "salary,department")
        """
    )
]

# Create a custom ReAct prompt template
template = """Answer the following questions about the dataframe as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Important Notes:
- The dataframe is available as 'df' in the python_repl tool
- Always use get_dataframe_info first to understand the data structure
- For complex queries, use python_repl with pandas operations
- Be precise with column names

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)

import os
os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1/"
os.environ["OPENAI_API_KEY"] = "test"

# Initialize the LLM
llm = ChatOpenAI(
    model="qwen/qwen3-4b-2507",
    temperature=0,
)

# Create the agent using create_react_agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

@traceable
def pandas_agent_query(query: str) -> str:
    """
    Function to query the pandas dataframe agent.
    """
    response = agent_executor.invoke({"input": query})
    return response['output']

# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Custom DataFrame Agent Demo")
    print("=" * 80)

    # query = "Total salary expenditure by department"
    # response = agent_executor.invoke({"input": query})
    # print(f"\nFinal Answer: {response['output']}\n")

    query = "Total salary expenditure by department"
    response = pandas_agent_query(query)
    print(f"\nFinal Answer: {response}\n")



    # # Example queries
    # queries = [
    #     "What are the columns in the dataframe?",
    #     "What is the average salary by department?",
    #     "Who has the highest salary?",
    #     "How many people are in each department?",
    #     "What is the average age of employees in Engineering department?"
    # ]
    
    # for query in queries:
    #     print(f"\n{'='*80}")
    #     print(f"Query: {query}")
    #     print(f"{'='*80}\n")
        
    #     try:
    #         response = agent_executor.invoke({"input": query})
    #         print(f"\nFinal Answer: {response['output']}\n")
    #     except Exception as e:
    #         print(f"Error: {str(e)}\n")
        
    #     print("-" * 80)

    # # Interactive mode
    # print("\n" + "="*80)
    # print("Interactive Mode - Type 'quit' to exit")
    # print("="*80 + "\n")
    
    # while True:
    #     user_input = input("\nYour question: ")
    #     if user_input.lower() in ['quit', 'exit', 'q']:
    #         break
        
    #     try:
    #         response = agent_executor.invoke({"input": user_input})
    #         print(f"\nAnswer: {response['output']}")
    #     except Exception as e:
    #         print(f"Error: {str(e)}")