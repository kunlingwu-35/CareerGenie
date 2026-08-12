import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd 

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

    messages = [
        {
            "role": "system",
            "content": """
You are Career Genie, a friendly and supportive career and education chatbot for students.

Help students explore careers, college and university majors, education pathways, skills, interests, and future goals.

Ask users about their favourite subjects, hobbies, strengths, interests, goals, and preferred work environment.

When recommending careers:
- Suggest 3 to 5 possible careers.
- Explain why each career may fit the user's interests.
- Suggest related majors or education pathways.
- Ask a helpful follow-up question.

If a user does not know what career they want, ask them questions about themselves before making recommendations.

Keep your responses friendly, clear, supportive, and easy for students to understand.

Do not tell users that there is only one correct career for them.

If you discuss salaries, admission requirements, university programs, or job outlooks, remind users that information can change and should be verified using reliable or official sources.

Stay focused on careers, education, majors, skills, and future planning.
"""
        }
    ]

    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    response = client.chat_completion(
        messages,
        max_tokens=700,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()


with gr.Blocks() as app:

    gr.Markdown(
        """
    Career Genie

    Find your path. Explore your future.

CareerCompass helps students explore careers, majors, and educational pathways based on their interests, strengths, favourite subjects, and goals.

Not sure where to begin? Try one of the questions below!
"""
    )

    chatbot = gr.ChatInterface(
        fn=respond,
        examples=[
            "I don't know what career I want.",
            "I love biology and helping people. What careers could fit me?",
            "I like coding and art. What careers combine both?",
            "What majors should I explore if I enjoy math?",
            "I like technology but also want to work with people."
        ]
    )


app.launch()