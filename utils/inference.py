from llm_config import SQL_AGENT_PREFIX, MODEL_ID, PIPELINE_INFERENCE_ARGS, bnb_config
from config import SQL_DATABASE_PATH
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain_huggingface import HuggingFacePipeline
from langchain.agents import initialize_agent


db = SQLDatabase.from_uri(f"sqlite:///{SQL_DATABASE_PATH}")


LLM_PIPE = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
    task="text-generation",
    device="auto",
    pipeline_kwargs=PIPELINE_INFERENCE_ARGS,
    model_kwargs={"quantization_config": bnb_config, "device_map": "auto"},
)

toolkit = SQLDatabaseToolkit(db=db, llm=LLM_PIPE)
tools = toolkit.get_tools()

agent_chain = initialize_agent(
    llm=LLM_PIPE,
    tools=tools,
    verbose=True,
    early_stopping_method="generate",
    handle_parsing_errors=True,
    max_iterations=3,
    return_intermediate_steps=True,
    agent_kwargs={
        "prefix": SQL_AGENT_PREFIX
    },  # comment this line to use zero-shot default prompt
)

if __name__ == "__main__":
    question = (
        "What is the expected result in the next match between Colombia and Paraguay?"
    )
    result = agent_chain.invoke({"input": question})
    print(result["output"])
