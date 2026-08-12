import convert_cip_spreadsheet_to_text

import gradio as gr
from huggingface_hub import InferenceClient
import pandas as pd 

#copy/paste from previous Semantic search assignment
#!pip install -q sentence-transformers
from sentence_transformers import SentenceTransformer  #turns sentences into vectors
import torch

with open("occupation_file", "r", encoding="utf-8") as file:
  # Read the entire contents of the file and store it in a variable
  occupation_text = file.read()

def preprocess_text(text):
  # Strip extra whitespace from the beginning and the end of the text
  cleaned_text = text.strip()

  # Split the cleaned_text by every newline character (\n)
  chunks = cleaned_text.split("\n")

  # Create an empty list to store cleaned chunks
  cleaned_chunks = []

  # Write your for-in loop below to clean each chunk and add it to the cleaned_chunks list
  # This is only one way scholars may write this, but there are other ways!
  for chunk in chunks:
    stripped_chunk = chunk.strip()
    if len(stripped_chunk) > 0:
      cleaned_chunks.append(stripped_chunk)

  # ===== SPICY CHALLENGE: LIST COMPREHENSION =====
  # The if chunk.strip() conditional is truthy if the string is not empty
  # cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

  # Print cleaned_chunks
  print(cleaned_chunks)

  # Print the length of cleaned_chunks
  print(len(cleaned_chunks))

  # Return the cleaned_chunks
  return cleaned_chunks

cleaned_chunks = preprocess_text(occupation_text)

    # Load the pre-trained embedding model that converts text to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_chunks):
  # Convert each text chunk into a vector embedding and store as a tensor
  chunk_embeddings = model.encode(cleaned_chunks, convert_to_tensor=True) # Replace ... with the text_chunks list

  # Print the chunk embeddings
  print(chunk_embeddings)

  # Print the shape of chunk_embeddings
  print(chunk_embeddings.shape)


  # Return the chunk_embeddings
  return chunk_embeddings

# Call the create_embeddings function and store the result in a new chunk_embeddings variable
chunk_embeddings = create_embeddings(cleaned_chunks) # Complete this line


# Define a function to find the most relevant text chunks for a given query, chunk_embeddings, and text_chunks
def get_top_chunks(query, chunk_embeddings, text_chunks):
  # Convert the query text into a vector embedding
  query_embedding = model.encode(query, convert_to_tensor = True) # Complete this line

  # Normalize the query embedding to unit length for accurate similarity comparison
  query_embedding_normalized = query_embedding / query_embedding.norm()

  # Normalize all chunk embeddings to unit length for consistent comparison
  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  # Calculate cosine similarity between all chunks and the query using matrix multiplication
  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) # Complete this line

  # Print the similarities
  print(similarities)

  # Find the indices of the 3 chunks with highest similarity scores
  top_indices = torch.topk(similarities, k=3).indices

  # Print the top indices
  print(top_indices)

  # Create an empty list to store the most relevant chunks
  top_chunks = []

  # Loop through the top indices and retrieve the corresponding text chunks
  for i in top_indices:
    chunk = text_chunks[i]
    top_chunks.append(chunk)

  # Return the list of most relevant chunks
  return top_chunks




# - -------------------- SOHA's code are below----------------
# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")


def respond(message, history):

    occupation_chunks = get_top_chunks(message, chunk_embeddings, cleaned_chunks)
    
    messages = [
        {
            "role": "system",
            "content": f"""
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

Stay focused on careers, education, majors, skills, and future planning. Provide your answers based on the following context {occupation_chunks}. Ask follow-up questions related the prompt. 
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


css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

.hero {
    background: linear-gradient(135deg, #dff5ec, #e3efff);
    padding: 40px;
    border-radius: 28px;
    text-align: center;
    margin-bottom: 20px;
}

.hero h1 {
    font-size: 46px;
    margin-bottom: 5px;
}

.card {
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #d8e2ec;
    min-height: 145px;
}

.quiz-box {
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #d8e2ec;
    margin-top: 20px;
}

.resource-box {
    padding: 18px;
    border-radius: 16px;
    border: 1px solid #d8e2ec;
    margin-bottom: 12px;
}

.big-text * {
    font-size: 110% !important;
}

.grayscale-mode {
    filter: grayscale(100%);
}
"""


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="emerald"
)


with gr.Blocks(
    theme=theme,
    css=css
) as app:

    gr.HTML(
        """
        <div class="hero">
            <div style="font-size:55px;">🧞‍♀️</div>

            <h1>Career Genie</h1>

            <h3>Your future, one question at a time.</h3>

            <p>
                Explore careers, majors, and education pathways
                based on your interests, strengths, and goals.
            </p>
        </div>
        """
    )


    gr.Markdown("## ♿ Accessibility & Display")


    with gr.Row():

        dark_button = gr.Button("🌙 Dark Mode")

        light_button = gr.Button("☀️ Light Mode")

        grayscale_button = gr.Button("◐ Grayscale")

        text_button = gr.Button("A+ Larger Text")


    gr.Markdown("## ✨ Explore Career Genie")


    with gr.Row():

        gr.HTML(
            """
            <div class="card">
                <h2>💼</h2>
                <h3>Explore Careers</h3>
                <p>Discover careers that match your interests and strengths.</p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="card">
                <h2>🎓</h2>
                <h3>Explore Majors</h3>
                <p>Learn about majors connected to different career paths.</p>
            </div>
            """
        )

        gr.HTML(
            """
            <div class="card">
                <h2>🧭</h2>
                <h3>Find Your Path</h3>
                <p>Not sure where to begin? Let Career Genie guide you.</p>
            </div>
            """
        )


    gr.Markdown("## 💬 Chat with Career Genie")


    chatbot = gr.ChatInterface(
        fn=respond,
        examples=[
            "I don't know what career I want.",
            "I love biology and helping people.",
            "I like coding and art.",
            "What majors should I explore if I like math?",
            "I like technology but also want to work with people."
        ]
    )


    gr.HTML(
        """
        <div class="quiz-box">

            <h2>🧠 Career Personality Quiz</h2>

            <p>
                Answer a few questions about your interests,
                strengths, and work preferences to discover
                careers that may fit you.
            </p>

            <p><strong>Quiz feature coming next ✨</strong></p>

        </div>
        """
    )


    gr.Markdown("## 🔗 Career Resources")


    gr.HTML(
        """
        <div class="resource-box">

            <h3>🇨🇦 Government of Canada Job Bank</h3>

            <p>
                Research careers, wages, requirements,
                and labour-market information.
            </p>

            <a href="https://www.jobbank.gc.ca/" target="_blank">
                Learn More →
            </a>

        </div>


        <div class="resource-box">

            <h3>🎓 Ontario Universities</h3>

            <p>
                Explore university programs and majors.
            </p>

            <a href="https://www.ouinfo.ca/" target="_blank">
                Explore Programs →
            </a>

        </div>


        <div class="resource-box">

            <h3>🏫 Ontario Colleges</h3>

            <p>
                Explore college programs and pathways.
            </p>

            <a href="https://www.ontariocolleges.ca/" target="_blank">
                Explore Colleges →
            </a>

        </div>
        """
    )


    dark_button.click(
        fn=None,
        js="""
        () => {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        }
        """
    )


    light_button.click(
        fn=None,
        js="""
        () => {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        }
        """
    )


    grayscale_button.click(
        fn=None,
        js="""
        () => {
            document.body.classList.toggle('grayscale-mode');
        }
        """
    )


    text_button.click(
        fn=None,
        js="""
        () => {
            document.body.classList.toggle('big-text');
        }
        """
    )


app.launch()