from llm_config import (
    SQL_AGENT_PROMPT,
    MODEL_ID,
    PIPELINE_INFERENCE_ARGS,
    bnb_config,
    parse_output,
)
from config import SQL_DATABASE_PATH
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_huggingface import HuggingFacePipeline
import sys

sys.path.append("../")

db = SQLDatabase.from_uri(f"sqlite:///{SQL_DATABASE_PATH}")


LLM_PIPE = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
    task="text-generation",
    device="auto",
    pipeline_kwargs=PIPELINE_INFERENCE_ARGS,
    model_kwargs={"quantization_config": bnb_config, "device_map": "auto"},
)


AGENT_EXECUTOR = create_sql_agent(
    LLM_PIPE,
    db=db,
    prompt=SQL_AGENT_PROMPT,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # set to True to see more details
    max_iterations=1,  # improve throughtput as reducing iterations
    agent_executor_kwargs={
        "return_intermediate_steps": True,
        "handle_parsing_errors": True,
    },
)


if __name__ == "__main__":
    question = """What is the goal average from Colombia in a match against Paraguay?"""
    result = AGENT_EXECUTOR.invoke({"input": question})
    if "Agent stopped due to iteration limit" in result["output"]:
        output = parse_output(result["intermediate_steps"])
    print(output)
