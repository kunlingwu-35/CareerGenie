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

#add CIP info to occupation_file
df = pd.read_excel("CIP2020_SOC2018_Crosswalk.xlsx")
cip_text = df.to_string(index=False)
occupation_text = occupation_text + "\n" + cip_text

# #add college major into occupation_file
# college_df = pd.read_csv("Most-Recent-Cohorts-Field-of-Study.csv")
# college_text = college_df.to_string(index=False)
# occupation_text = occupation_text + "\n" + college_text

college_df = pd.read_csv("Most-Recent-Cohorts-Field-of-Study.csv")

college_df = college_df[
    [
        "INSTNM",
        "CIPCODE",
        "CIPDESC",
        "CREDDESC",
        "IPEDSCOUNT1",
        "IPEDSCOUNT2",
        "EARN_MDN_4YR"
    ]
]

college_chunks = []

for _, row in college_df.iterrows():

    text = (
        f"College: {row['INSTNM']} | "
        f"Major: {row['CIPDESC']} | "
        f"Degree: {row['CREDDESC']} | "
        f"CIP Code: {row['CIPCODE']} | "
        f"Graduates: {row['IPEDSCOUNT2']} | "
        f"Median earnings 4 years after completion: {row['EARN_MDN_4YR']}"
    )

    college_chunks.append(text)

print(college_chunks[:5])

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
all_chunks = cleaned_chunks + college_chunks

chunk_embeddings = create_embeddings(all_chunks) # Complete this line


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

    occupation_chunks = get_top_chunks(message, chunk_embeddings, all_chunks)
    
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


# -----------------------------
# QUIZ
# -----------------------------

def quiz_result(subject, work_style, interest, value):

    results = []

    if subject == "Science / Biology":
        results.append(" Healthcare, Biomedical Science, Biotechnology")

    elif subject == "Math":
        results.append(" Engineering, Data Science, Finance")

    elif subject == "Technology":
        results.append(" Computer Science, Engineering, Cybersecurity")

    elif subject == "Art / Design":
        results.append(" UX/UI Design, Architecture, Media")

    elif subject == "Business":
        results.append(" Marketing, Finance, Entrepreneurship")


    if interest == "Helping people":
        results.append(" Medicine, Nursing, Psychology, Education")

    elif interest == "Building things":
        results.append(" Engineering, Product Design, Architecture")

    elif interest == "Solving problems":
        results.append(" Engineering, Consulting, Data Science")

    elif interest == "Creating":
        results.append(" Design, Media, Marketing, UX/UI")

    elif interest == "Working with technology":
        results.append(" Software, AI, Cybersecurity, Engineering")


    if work_style == "Working with people":
        results.append(" Careers involving teamwork, communication, or clients")

    elif work_style == "Working independently":
        results.append(" Research, programming, writing, design")

    elif work_style == "A mix of both":
        results.append(" Careers with both independent work and collaboration")


    if value == "Helping others":
        results.append(" Healthcare, education, public service")

    elif value == "Creativity":
        results.append(" Design, media, marketing")

    elif value == "High earning potential":
        results.append(" Engineering, technology, finance")

    elif value == "Work-life balance":
        results.append(" Explore careers with flexible work environments")

    elif value == "Making an impact":
        results.append(" Healthcare, engineering, sustainability, nonprofit work")


    if not results:
        return "Try selecting a few options so Career Genie can suggest possible pathways."


    final_text = "###  Your Career Genie Matches\n\n"

    for item in results:
        final_text += "- " + item + "\n"

    final_text += "\nUse these results as a starting point, then ask Career Genie about any career that interests you!"

    return final_text


# -----------------------------
# SAVED CAREERS
# -----------------------------

def save_career(career, saved):

    if saved is None:
        saved = []

    career = career.strip()

    if career == "":
        return saved, saved

    if career not in saved:
        saved.append(career)

    return saved, saved


def clear_saved():
    return [], []


# -----------------------------
# CUSTOM CSS
# -----------------------------

css = """

.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

body {
    background: linear-gradient(135deg, #eef9f5, #edf4ff);
}

#hero {
    background: linear-gradient(135deg, #cfeee2, #d8e8ff);
    border-radius: 30px;
    padding: 45px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

#hero h1 {
    font-size: 48px;
    margin-bottom: 5px;
}

#hero h2 {
    font-weight: 400;
    margin-top: 5px;
}

#accessibility {
    background: white;
    padding: 18px;
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid #e5e7eb;
}

.feature-card {
    background: white;
    border-radius: 20px;
    padding: 20px;
    text-align: center;
    border: 1px solid #e5e7eb;
    min-height: 145px;
}

#chat-section {
    background: white;
    border-radius: 22px;
    padding: 18px;
    margin-top: 20px;
}

#quiz-section {
    background: #f7fbff;
    border-radius: 22px;
    padding: 22px;
    margin-top: 25px;
}

#saved-section {
    background: #fffdf4;
    border-radius: 22px;
    padding: 22px;
    margin-top: 25px;
}

#resources-section {
    background: #f3faf6;
    border-radius: 22px;
    padding: 22px;
    margin-top: 25px;
}

.footer {
    text-align: center;
    opacity: 0.7;
    font-size: 13px;
    margin-top: 25px;
}

.dark-mode {
    background: #121212 !important;
    color: white !important;
}

.dark-mode .gradio-container {
    background: #121212 !important;
    color: white !important;
}

.grayscale-mode {
    filter: grayscale(100%) !important;
}

.large-text {
    font-size: 120% !important;
}

.color-friendly {
    background: #fff3c4 !important;
    color: #003366 !important;
}

"""


# -----------------------------
# JAVASCRIPT FOR ACCESSIBILITY
# -----------------------------

js = """

function careerLightMode() {
    document.body.classList.remove(
        "dark-mode",
        "grayscale-mode",
        "color-friendly"
    );
}

function careerDarkMode() {
    document.body.classList.remove(
        "grayscale-mode",
        "color-friendly"
    );

    document.body.classList.add("dark-mode");
}

function careerGrayscale() {
    document.body.classList.toggle("grayscale-mode");
}

function careerColorFriendly() {
    document.body.classList.remove("dark-mode");
    document.body.classList.toggle("color-friendly");
}

function careerLargeText() {
    document.body.classList.toggle("large-text");
}

function careerReset() {
    document.body.classList.remove(
        "dark-mode",
        "grayscale-mode",
        "color-friendly",
        "large-text"
    );
}

"""


# -----------------------------
# APP
# -----------------------------

with gr.Blocks(
    css=css,
    js=js,
    theme=gr.themes.Soft()
) as app:


    # HERO / BANNER

    gr.HTML(
        """
        <div id="hero">

            <img src="file=Career (1).png" 
                style="width:200px; height:200px; object-fit:contain;">

            <div style="font-size:60px;">
                
            </div>

            <h1>Career Genie</h1>

            <h2>Your future, one question at a time.</h2>

            <p>
                Discover careers, majors, educational pathways,
                and possibilities based on your interests,
                strengths, personality, and goals.
            </p>

        </div>
        """
    )


    # ACCESSIBILITY

    with gr.Column(elem_id="accessibility"):

        gr.Markdown(
            """
###  Accessibility & Display

Customize Career Genie so the app is easier and more comfortable for you to use.
"""
        )

        with gr.Row():

            light_button = gr.Button(" Light")
            dark_button = gr.Button(" Dark")
            grayscale_button = gr.Button(" Grayscale")
            color_button = gr.Button(" Colour-Friendly")
            text_button = gr.Button(" Larger Text")
            reset_button = gr.Button(" Reset")


    light_button.click(
        fn=None,
        js="careerLightMode"
    )

    dark_button.click(
        fn=None,
        js="careerDarkMode"
    )

    grayscale_button.click(
        fn=None,
        js="careerGrayscale"
    )

    color_button.click(
        fn=None,
        js="careerColorFriendly"
    )

    text_button.click(
        fn=None,
        js="careerLargeText"
    )

    reset_button.click(
        fn=None,
        js="careerReset"
    )


    # FEATURE CARDS

    gr.Markdown("##  Explore Your Future")


    with gr.Row():

        gr.Markdown(
            """
###  Explore Careers

Discover careers based on your interests, strengths, and goals.
            """,
            elem_classes="feature-card"
        )

        gr.Markdown(
            """
###  Explore Majors

Learn about college and university programs connected to different careers.
            """,
            elem_classes="feature-card"
        )

        gr.Markdown(
            """
###  Find Your Path

Not sure what you want to do yet? Career Genie can help you narrow down your options.
            """,
            elem_classes="feature-card"
        )


    # CHATBOT

    with gr.Column(elem_id="chat-section"):

        gr.Markdown(
            """
##  Chat with Career Genie

Tell Career Genie about your interests, strengths, favourite subjects, or future goals.
"""
        )

        chatbot = gr.ChatInterface(
            fn=respond,
            examples=[
                "I don't know what career I want.",
                "I love biology and helping people. What careers could fit me?",
                "I enjoy coding and art. What careers combine both?",
                "What majors should I explore if I enjoy math?",
                "I like technology but also want to work with people."
            ]
        )


    # PERSONALITY QUIZ

    with gr.Column(elem_id="quiz-section"):

        gr.Markdown(
            """
##  Career Personality & Interests Quiz

Answer a few questions and Career Genie will suggest possible paths to explore.
"""
        )

        subject = gr.Dropdown(
            choices=[
                "Science / Biology",
                "Math",
                "Technology",
                "Art / Design",
                "Business"
            ],
            label="What subject do you enjoy most?"
        )

        work_style = gr.Radio(
            choices=[
                "Working with people",
                "Working independently",
                "A mix of both"
            ],
            label="How do you prefer to work?"
        )

        interest = gr.Dropdown(
            choices=[
                "Helping people",
                "Building things",
                "Solving problems",
                "Creating",
                "Working with technology"
            ],
            label="Which activity sounds most like you?"
        )

        value = gr.Dropdown(
            choices=[
                "Helping others",
                "Creativity",
                "High earning potential",
                "Work-life balance",
                "Making an impact"
            ],
            label="What matters most to you in a future career?"
        )

        quiz_button = gr.Button(" Find My Career Matches")

        quiz_output = gr.Markdown()


        quiz_button.click(
            fn=quiz_result,
            inputs=[
                subject,
                work_style,
                interest,
                value
            ],
            outputs=quiz_output
        )


    # SAVE CAREERS

    with gr.Column(elem_id="saved-section"):

        gr.Markdown(
            """
##  Save Careers

Found something interesting?

Type the career below and save it so you can keep track of careers you want to research later.
"""
        )

        saved_state = gr.State([])

        career_input = gr.Textbox(
            label="Career to save",
            placeholder="Example: Biomedical Engineer"
        )

        with gr.Row():

            save_button = gr.Button(" Save Career")

            clear_button = gr.Button(" Clear Saved Careers")


        saved_output = gr.JSON(
            label="My Saved Careers"
        )


        save_button.click(
            fn=save_career,
            inputs=[
                career_input,
                saved_state
            ],
            outputs=[
                saved_state,
                saved_output
            ]
        )


        clear_button.click(
            fn=clear_saved,
            outputs=[
                saved_state,
                saved_output
            ]
        )


    # LINKS / RESOURCES

    with gr.Column(elem_id="resources-section"):

        gr.Markdown(
            """
##  Continue Exploring

Career Genie can help you discover possibilities, but you should always verify important career and education information using reliable sources.

### 🇨🇦 Government of Canada Job Bank

Research occupations, wages, skills, requirements, and job outlooks.

https://www.jobbank.gc.ca/

###  Ontario Universities Info

Explore Ontario university programs, prerequisites, and admission information.

https://www.ouinfo.ca/

###  Ontario Colleges

Explore college programs and pathways across Ontario.

https://www.ontariocolleges.ca/

###  How to use this section

If Career Genie recommends **Biomedical Engineering**, for example, you can search that career on Job Bank and then use Ontario Universities Info to explore related university programs.
"""
        )


    # FOOTER

    gr.Markdown(
        """
Career Genie provides general educational guidance. Career information, salaries,
admission requirements, and programs may change, so always verify important
information using official sources.
        """,
        elem_classes="footer"
    )


app.launch()