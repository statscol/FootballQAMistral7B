import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
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
    "max_new_tokens": 256,
    "do_sample": True,
    "eos_token_id": TERMINATORS,
    "temperature": 0.1,
    "repetition_penalty": 1.1,
    "top_p": 0.6,
}


# modifying default template from
# https://github.com/langchain-ai/langchain/blob/0cd3f9336164b0971625f19064d07fb08577bf40/libs/community/langchain_community/agent_toolkits/sql/base.py#L163


SQL_AGENT_PROMPT = PromptTemplate(
    input_variables=["agent_scratchpad", "input"],
    partial_variables={
        "table_metadata": """
            - The table 'goals' contains all the goals scored in official FIFA matches between two national teams:
                - date: date the match took place
                - home_team: national team which was considered local team.
                - away_team: national team which was considered away team.
                - scorer: name of the player who scored a goal.
                - team: national team of the player who scored the goal in the 'scorer' column.
                - minute: minute in which the goal was scored.
                - penalty: boolean whether or not the goal scored form the 'scorer' was a penalty.
                - own_goal: whether or not the goal scored was an own goal.
            - The table 'matches' stores the results from official FIFA matches:
                - date: date the match took place.
                - home_team: national team which was considered local team.
                - away_team: national team which was considered away team.
                - home_score: number of goals scored by the home team.
                - away_score: number of goals scored by the away team.
                - tournament:  tournament the match took place in the column, e.g 'Friendly', 'FIFA World Cup','Copa América','UEFA Euro'.
                - city: city in which the match was played.
                - country: country in which the match was played.
                - neutral: boolean used to indicate if the match was played in neutral territory
                     (True if not in any of the national teams land).
            - The 'players' table is the latest data available for player rankings and characteristics.
             This table can be linked to other tables using the 'nationality_name':
                - short_name: player short name.
                - long_name: player long name.
                - height_cm: height of the player.
                - nationality_name: national team of the player.
                - age: player age in years
                - pace: player rating for speed and acceleration from 1 to 100.
                - shooting: player rating for shooting from 1 to 100.
                - passing: player rating for pass accuracy from 1 to 100.
                - defending: player rating for defending from 1 to 100.
                - dribbling: player rating for his ability and agility to dribble from 1 to 100.
                - physic: player rating for jumping, stamina and strength from 1 to 100.
                - overall: average score of player stats.
            """,
        "tools": """
            sql_db_query - Input to this tool is a detailed and correct SQL query, output is a result from the database.
            If the query is not correct, an error message will be returned.
            If an error is returned, rewrite the query, check the query, and try again.
            If you encounter an issue with Unknown column 'xxxx' in 'field list',
            use sql_db_schema to query the correct table fields.
            sql_db_schema - Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
            Be sure that the tables actually exist by calling sql_db_list_tables first!
            Example Input: table1, table2, table3 also make sure the fields exist in a table.
            sql_db_list_tables - Input is an empty string, output is a comma-separated list of tables in the database.
            sql_db_query_checker - Use this tool to double check if your query is correct before executing it.
            Always use this tool before executing a query with sql_db_query!""",
        "tool_names": "sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker",
    },
    template="""Answer the following questions as best you can. You have access to the following tools:{tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do clearly.
                Action: the action to take, should be one of [{tool_names}].
                Also you have the following description of the fields for every table in the database: {table_metadata}
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question. If a final answer has been reached, finish.

                Begin!

                Question: {input}
                Thought:{agent_scratchpad}
                """,
)


def prepare_prompt(prompt: str):
    """add init and end tokens to a manual prompt"""
    return f"{MODEL_INIT_TOKEN} {prompt} {MODEL_END_TOKEN}"


def parse_output(interm_steps: list, return_full_text: bool = False):
    """alternative to force early_stopping_method in create_sql_agent"""
    return (
        str(interm_steps[-1][0])
        .split("Begin!" if return_full_text else "Final Answer:")[-1]
        .strip()
    )
