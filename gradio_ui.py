import os

import google.generativeai as genai
from dotenv import load_dotenv, find_dotenv
import gradio as gr

load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

model = genai.GenerativeModel()


def handle_user_query(message: str, chat_history: list[tuple]) -> tuple:
    chat_history += [[message, None]]
    return '', chat_history


def generate_chat_history(chat_history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    formatted_chat_history = []
    if len(chat_history) == 0:
        return formatted_chat_history
    for ch in chat_history:
        formatted_chat_history.append(
            {
                "role": "user",
                "parts": [ch[0]]
            }
        )
        formatted_chat_history.append(
            {
                "role": "model",
                "parts": [ch[1]]
            }
        )
    return formatted_chat_history


def handle_gemini_response(chat_history: list[tuple]):
    query = chat_history[-1][0]
    history = generate_chat_history(chat_history[:-1])
    chat = model.start_chat(history=history)
    response = chat.send_message(query, stream=True)
    print(response)
    chat_history[-1][1] = ''
    for chunk in response:
        chat_history[-1][1] += chunk.text
        yield chat_history
    return None


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        label='Chat with Gemini',
        bubble_full_width=False,

    )
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(
        handle_user_query,
        [msg, chatbot],
        [msg, chatbot]
    ).then(
        handle_gemini_response,
        [chatbot],
        [chatbot]
    )

demo.queue()
