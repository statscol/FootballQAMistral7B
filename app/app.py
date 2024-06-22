import gradio as gr
from config import DEFAULT_BOT_MESSAGE
from llm_config import PIPELINE_INFERENCE_ARGS
from inference import agent_executor


# adapted from https://github.com/gradio-app/gradio/issues/7925#issuecomment-2041571560
def format_history(msg: str, history: list[list[str, str]], system_prompt: str):
    chat_history = [{"role": "system", "content": system_prompt}]
    for query, response in history:
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": response})
    chat_history.append({"role": "user", "content": msg})
    return chat_history


def generate_response(
    msg: str,
    history: list[list[str, str]],
    system_prompt: str,
    top_k: int,
    top_p: float,
    temperature: float,
):
    chat_history = format_history(msg, history, system_prompt)
    PIPELINE_INFERENCE_ARGS["top_k"] = top_k
    PIPELINE_INFERENCE_ARGS["top_p"] = top_p
    PIPELINE_INFERENCE_ARGS["temperature"] = temperature
    print(PIPELINE_INFERENCE_ARGS)
    response = agent_executor.run(msg)
    return response


chatbot = gr.ChatInterface(
    generate_response,
    chatbot=gr.Chatbot(
        value=[(None, DEFAULT_BOT_MESSAGE)],
        height="64vh",
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
            value=40,
            info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)",
        ),
        gr.Slider(
            0.0,
            1.0,
            label="top_p",
            value=0.9,
            info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)",
        ),
        gr.Slider(
            0.1,
            1.0,
            label="temperature",
            value=0.1,
            info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.1)",
        ),
    ],
    title="FootballQA",
    submit_btn="⬅ Send",
    retry_btn="🔄 Regenerate Response",
    undo_btn="↩ Delete Previous",
    clear_btn="🗑️ Clear Chat",
    cache_examples=True,
    css="footer {visibility: hidden}",
)

if __name__ == "__main__":
    chatbot.queue().launch(server_name="0.0.0.0", server_port=9090)
