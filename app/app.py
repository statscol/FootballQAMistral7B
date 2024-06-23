import gradio as gr
import sys

sys.path.append("utils")
from config import DEFAULT_BOT_MESSAGE
from llm_config import PIPELINE_INFERENCE_ARGS, parse_output
from inference import AGENT_EXECUTOR


# adapted from https://github.com/gradio-app/gradio/issues/7925#issuecomment-2041571560
# currently not used in a MemoryBuffer
def format_history(msg: str, history: list[list[str, str]], system_prompt: str):
    chat_history = [{"role": "system", "content": system_prompt}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history


def generate_response(
    msg: str,
    chat_history: list,
    top_k: int,
    top_p: float,
    temperature: float,
    return_full_thought: bool,
):
    PIPELINE_INFERENCE_ARGS["top_k"] = top_k
    PIPELINE_INFERENCE_ARGS["top_p"] = top_p
    PIPELINE_INFERENCE_ARGS["temperature"] = temperature
    response = AGENT_EXECUTOR.invoke({"input": msg})
    output = (
        parse_output(response["intermediate_steps"], return_full_thought)
        if "Agent stopped due to iteration limit" in response["output"]
        else response["output"].strip()
    )
    return output


chatbot = gr.ChatInterface(
    generate_response,
    chatbot=gr.Chatbot(
        value=[(None, DEFAULT_BOT_MESSAGE)],
        height="64vh",
        avatar_images=[
            "https://cdn-icons-png.flaticon.com/512/9385/9385289.png",
            "https://cdn-icons-png.flaticon.com/512/8649/8649607.png",
        ],
        render_markdown=True,
    ),
    examples=[
        ["What is the win probability of Colombia vs Paraguay?"],
        [
            "What is the expected number of goals in a match between Argentina and Brazil?"
        ],
    ],
    additional_inputs=[
        gr.Slider(
            0.0,
            100.0,
            label="top_k",
            value=10,
            info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 10)",
        ),
        gr.Slider(
            0.0,
            1.0,
            label="top_p",
            value=0.6,
            info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.6)",
        ),
        gr.Slider(
            0.1,
            1.0,
            label="temperature",
            value=0.2,
            info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.2)",
        ),
        gr.Checkbox(
            label="return_full_thought",
            value=False,
            info="When the agent reaches max iterations you can return the intermediate steps and the full thought behind the answer",
        ),
    ],
    title="FootballQA",
    submit_btn="⬅ Send",
    retry_btn="🔄 Regenerate Response",
    undo_btn="↩ Delete Previous",
    clear_btn="🗑️ Clear Chat",
    cache_examples=False,
    css="footer {visibility: hidden}",
)

if __name__ == "__main__":
    chatbot.queue().launch(server_name="0.0.0.0", server_port=9090)
