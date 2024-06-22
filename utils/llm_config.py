import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from config import MODEL_ID, MODEL_INIT_TOKEN, MODEL_END_TOKEN
from langchain.prompts import PromptTemplate

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Tokenizer & Model
# You must request access to the checkpoints
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)

TERMINATORS = [TOKENIZER.eos_token_id, TOKENIZER.convert_tokens_to_ids("[/INST]")]

PIPELINE_INFERENCE_ARGS = {
    "max_new_tokens": 1024,
    "eos_token_id": TERMINATORS,
    "do_sample": True,
    "temperature": 0.1,
    "top_p": 0.9,
}

LLM_PIPE = HuggingFacePipeline.from_model_id(
    model_id=MODEL_ID,
    task="text-generation",
    device="auto",
    pipeline_kwargs=PIPELINE_INFERENCE_ARGS,
    model_kwargs={"quantization_config": bnb_config, "device_map": "auto"},
)

# modifying default template from https://github.com/langchain-ai/langchain/blob/0cd3f9336164b0971625f19064d07fb08577bf40/libs/community/langchain_community/agent_toolkits/sql/base.py#L163
SQL_AGENT_PROMPT = PromptTemplate(
    input_variables=["agent_scratchpad", "input"],
    partial_variables={
        "table_metadata": """
            the table 'goals' contains all the goals scored in official FIFA matches between two national teams.
            Also in the table 'matches' you can find the results from all the matches, both tables are linked using the 'id' column.""",
        "tools": """
            sql_db_query - Input to this tool is a detailed and correct SQL query, output is a result from the database. make sure the values queried are in
            lower case for convenience and also remove accented letters if needed.If the query is not correct, an error message will be returned.
            If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list',
            use sql_db_schema to query the correct table fields.
            sql_db_schema - Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
            Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3 
            sql_db_list_tables - Input is an empty string, output is a comma-separated list of tables in the database.
            sql_db_query_checker - Use this tool to double check if your query is correct before executing it. 
            Always use this tool before executing a query with sql_db_query!""",
        "tool_names": "sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker",
    },
    template="""Answer the following questions as best you can. Always use the data you can pull from the tables. You have access to the following tools:{tools}

                Information about the database: {table_metadata}

                Use the following format:
                
                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question
                        
                Begin!
                
                Question: {input}
                Thought:{agent_scratchpad}""",
)


def prepare_prompt(prompt: str):
    """add init and end tokens to a manual prompt"""
    return f"{MODEL_INIT_TOKEN} {prompt} {MODEL_END_TOKEN}"
