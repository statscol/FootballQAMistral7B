from llm_config import SQL_AGENT_PROMPT, MODEL_ID, PIPELINE_INFERENCE_ARGS, bnb_config
from config import SQL_DATABASE_PATH
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser


db = SQLDatabase.from_uri(f"sqlite:///{SQL_DATABASE_PATH}")


LLM_PIPE = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
    task="text-generation",
    device="auto",
    pipeline_kwargs=PIPELINE_INFERENCE_ARGS,
    model_kwargs={"quantization_config": bnb_config, "device_map": "auto"},
)


agent_executor = create_sql_agent(
    LLM_PIPE,
    db=db,
    prompt=SQL_AGENT_PROMPT,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,  # set to True to see more details
    agent_executor_kwargs={
        "return_intermediate_steps": True,
        "handle_parsing_errors": True,
    },
)

if __name__ == "__main__":
    question = """What is the expected result from the next match between Colombia and Paraguay in Copa América?
      use the data between 2010 and 2022 and direct encounters between the two, also return Colombia's probability of winning the match """
    result = agent_executor.invoke({"input": question})["output"]
    result = (
        agent_executor.invoke({"input": question})["output"]
        or "Sorry, I Couldn't process your question"
    )
    print(result)
