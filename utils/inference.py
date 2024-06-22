from llm_config import LLM_PIPE, SQL_AGENT_PROMPT
from config import SQL_DATABASE_PATH
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.agents.mrkl import prompt as react_prompt

db = SQLDatabase.from_uri(f"sqlite:///{SQL_DATABASE_PATH}")

# modifying default template from https://github.com/langchain-ai/langchain/blob/0cd3f9336164b0971625f19064d07fb08577bf40/libs/community/langchain_community/agent_toolkits/sql/base.py#L163


agent_executor = create_sql_agent(
    LLM_PIPE,
    db=db,
    prompt=SQL_AGENT_PROMPT,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    handle_sql_errors=True,
)


if __name__ == "__main__":
    question = """Tell me the expected result from the next match between Colombia and Paraguay in Copa América,
      use the data from the last 5 years and the average number of goals per team"""
    response = agent_executor.run(question)
    print(response)
