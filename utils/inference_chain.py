from llm_config import SQL_CHAIN_PROMPT, MODEL_ID, PIPELINE_INFERENCE_ARGS, bnb_config
from config import SQL_DATABASE_PATH, DEFAULT_ROWS_CONTEXT
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chains import create_sql_query_chain
from langchain_huggingface import HuggingFacePipeline


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

CHAIN_INFERENCE = create_sql_query_chain(
    LLM_PIPE, db, prompt=SQL_CHAIN_PROMPT, k=DEFAULT_ROWS_CONTEXT
)


if __name__ == "__main__":
    question = (
        "What is the expected result in the next match between Colombia and Paraguay?"
    )
    result = CHAIN_INFERENCE.invoke({"question": question})
    print(result)
