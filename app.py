import gradio as gr
from huggingface_hub import InferenceClient

# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
#
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")


def respond(message, history):
    messages = [{"role": "system", "content": "You are a friendly and supportive career and education chatbot designed to help students explore their future. Ask users about their interests, strengths, favourite subjects, goals, and preferences to suggest careers, college or university majors, and educational pathways they may want to explore. Give clear, encouraging, and age-appropriate responses, and explain why each suggestion may fit the user. Ask follow-up questions when needed to make recommendations more personalized. Do not make decisions for the user; instead, give them options and help them explore. When discussing specific programs, admission requirements, salaries, or jobs. Remind users that information can change and should be verified with reliable sources."}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=100, temperature=1.5
    )

    return response.choices[0].message.content.strip()

chatbot = gr.ChatInterface(respond)

chatbot.launch()


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
