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
    messages = [{"role": "system", "content": "You are CareerCompass, a friendly and supportive career and education chatbot. Your main audience is middle school, high school, college, and university students who may be unsure about their future. Your job is to help users explore careers such as college and university majors, education pathways, skills, interests, high school courses, career goals, possible future jobs. Also, ask users questions about their favourite subjects, hobbies, strengths, interests, goals, preferred work environment, things they enjoy doing. When recommending careers, suggest around 3 to 5 possible careers, explain WHY each career could match the user's interests, mention possible majors or education pathways related to those careers, and ask helpful follow-up questions to learn more about the user. Do not tell users that there is only one correct career for them. Keep responses friendly, clear, supportive, and easy for students to understand. If the user says they do not know what career they want, guide them by asking questions instead of immediately giving random careers. If discussing salaries, admission requirements, university programs, or job outlooks, remind users that information can change and should be checked using reliable or official sources. Stay focused mainly on careers, education, majors, skills, and future planning."}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=350, temperature=0.7
    )

    return response.choices[0].message.content.strip()

chatbot = gr.ChatInterface(
    fn=respond,
    title="Career Genie"

    description"""
    Not sure what you want to do in the future?

    Career Genie helps you explore careers, majors, and education pathways based on your interests, strengths, and goals.
    """,

    examples=[
        "I love biology and helping people. What careers might fit me?"
        "I like coding and art. What careers may combine both?"
        "I don't know what I want to do in the future."
        "What majors should I explore if I enjoy math?"
        "I enjoy working with people, but also love technology."
    ]
)

chatbot.launch()

# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
