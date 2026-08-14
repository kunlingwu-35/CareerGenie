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
:root {
    --bg-green: #edf8f4;
    --bg-blue: #eef4ff;
    --bg-purple: #f7f1ff;
    --card: #ffffff;
    --text: #24343d;
    --muted: #647680;
    --green: #4b9b89;
    --green-dark: #317565;
    --blue: #6688c5;
    --border: #dce8e5;
    --shadow: rgba(46, 74, 71, 0.10);
}

html,
body {
    min-height: 100%;
}

body {
    margin: 0;

    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(200, 240, 226, 0.85),
            transparent 32%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(207, 222, 255, 0.85),
            transparent 30%
        ),
        linear-gradient(
            145deg,
            var(--bg-green),
            var(--bg-blue),
            var(--bg-purple)
        ) !important;

    color: var(--text);

    transition:
        background 0.3s ease,
        color 0.3s ease;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding: 24px 18px 50px !important;
    background: transparent !important;
}

#hero {
    position: relative;
    overflow: hidden;

    background:
        linear-gradient(
            135deg,
            #d5f2e7,
            #dce9ff,
            #eee3ff
        );

    border-radius: 32px;

    padding: 42px 30px;

    margin-bottom: 22px;

    text-align: center;

    border:
        1px solid
        rgba(255, 255, 255, 0.85);

    box-shadow:
        0 18px 50px
        rgba(45, 76, 73, 0.13);
}

#hero::before {
    content: "";

    position: absolute;

    width: 250px;
    height: 250px;

    border-radius: 50%;

    right: -90px;
    top: -110px;

    background:
        rgba(255, 255, 255, 0.26);
}

#hero::after {
    content: "";

    position: absolute;

    width: 190px;
    height: 190px;

    border-radius: 50%;

    left: -70px;
    bottom: -100px;

    background:
        rgba(255, 255, 255, 0.20);
}

#hero h1 {
    position: relative;
    z-index: 2;

    font-size: 54px;

    margin-top: 8px;
    margin-bottom: 8px;

    line-height: 1;

    letter-spacing: -2px;

    color: #244148;
}

#hero h3 {
    position: relative;
    z-index: 2;

    margin-top: 8px;

    font-weight: 500;

    color: #536a78;
}

#hero p {
    position: relative;
    z-index: 2;

    color: #607482;
}

#hero img {
    position: relative;
    z-index: 2;

    border-radius: 24px !important;

    max-height: 170px !important;

    object-fit: contain !important;
}

#accessibility {
    background:
        rgba(255, 255, 255, 0.90);

    padding: 21px;

    border-radius: 23px;

    margin-bottom: 22px;

    border:
        1px solid
        var(--border);

    box-shadow:
        0 8px 28px
        rgba(45, 76, 72, 0.07);
}

.feature-card {
    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f9fcfb
        );

    border-radius: 21px;

    padding: 22px;

    border:
        1px solid
        var(--border);

    min-height: 145px;

    box-shadow:
        0 8px 24px
        rgba(45, 75, 72, 0.07);

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}

.feature-card:hover {
    transform:
        translateY(-4px);

    box-shadow:
        0 14px 32px
        rgba(45, 75, 72, 0.12);
}

#chat-section,
#quiz-section,
#saved-section,
#resources-section {
    border-radius: 24px;

    padding: 23px;

    margin-top: 23px;

    border:
        1px solid
        var(--border);

    box-shadow:
        0 10px 34px
        var(--shadow);
}

#chat-section {
    background:
        linear-gradient(
            145deg,
            #ffffff,
            #f9fcfd
        );
}

#quiz-section {
    background:
        linear-gradient(
            145deg,
            #f8fcff,
            #f3f8ff
        );
}

#saved-section {
    background:
        linear-gradient(
            145deg,
            #fffdf7,
            #faf8ef
        );
}

#resources-section {
    background:
        linear-gradient(
            145deg,
            #f7fcf9,
            #f7f9ff
        );
}

button {
    border-radius: 13px !important;

    font-weight: 700 !important;

    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease !important;
}

button:hover {
    transform:
        translateY(-1px);

    box-shadow:
        0 6px 14px
        rgba(45, 75, 72, 0.12);
}

.primary-button {
    background:
        linear-gradient(
            135deg,
            #4b9b89,
            #6688c5
        ) !important;

    color:
        white !important;

    border:
        none !important;
}

#quiz-output {
    background:
        rgba(255, 255, 255, 0.75);

    border-radius: 17px;

    padding: 15px;

    border:
        1px solid
        var(--border);
}

.footer {
    text-align: center;

    opacity: 0.72;

    font-size: 13px;

    margin-top: 28px;

    padding-bottom: 12px;
}

body.dark-mode {
    --card: #202833;
    --text: #edf3f7;
    --muted: #b8c4cc;
    --border: #3e4a56;
    --shadow: rgba(0, 0, 0, 0.28);

    background:
        radial-gradient(
            circle at 10% 8%,
            rgba(39, 86, 75, 0.60),
            transparent 31%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(49, 68, 112, 0.58),
            transparent 30%
        ),
        linear-gradient(
            145deg,
            #12171d,
            #171e27,
            #1d1924
        ) !important;
}

.dark-mode .gradio-container {
    color:
        #edf3f7 !important;
}

.dark-mode #hero {
    background:
        linear-gradient(
            135deg,
            #22483f,
            #293a57,
            #40324f
        );

    border-color:
        #465360;
}

.dark-mode #hero h1 {
    color:
        #f3f7f8;
}

.dark-mode #hero h3,
.dark-mode #hero p {
    color:
        #cad6de;
}

.dark-mode #accessibility,
.dark-mode .feature-card,
.dark-mode #chat-section,
.dark-mode #quiz-section,
.dark-mode #saved-section,
.dark-mode #resources-section,
.dark-mode #quiz-output {
    background:
        #202833 !important;

    color:
        #edf3f7 !important;

    border-color:
        #3e4a56 !important;
}

.dark-mode h1,
.dark-mode h2,
.dark-mode h3,
.dark-mode h4,
.dark-mode p,
.dark-mode label,
.dark-mode span {
    color:
        #edf3f7;
}

.dark-mode input,
.dark-mode textarea {
    background:
        #29333f !important;

    color:
        #edf3f7 !important;

    border-color:
        #465462 !important;
}

.grayscale-mode {
    filter:
        grayscale(100%);
}

.large-text {
    font-size:
        120% !important;
}

.large-text input,
.large-text textarea,
.large-text button {
    font-size:
        110% !important;
}

.color-friendly {
    background:
        linear-gradient(
            145deg,
            #fff4c7,
            #e9f3ff
        ) !important;
}

body.high-contrast {
    background:
        #ffffff !important;

    color:
        #000000 !important;
}

.high-contrast #accessibility,
.high-contrast .feature-card,
.high-contrast #chat-section,
.high-contrast #quiz-section,
.high-contrast #saved-section,
.high-contrast #resources-section {
    border:
        2px solid
        #000000 !important;
}

.high-contrast button {
    border:
        2px solid
        #000000 !important;
}

@media screen and (max-width: 700px) {
    .gradio-container {
        padding:
            12px
            9px
            35px
            !important;
    }

    #hero {
        padding:
            28px
            18px;
    }

    #hero h1 {
        font-size:
            42px;
    }

    .feature-card {
        min-height:
            auto;
    }
}
"""


js = """
function careerLightMode() {
    document.body.classList.remove(
        "dark-mode"
    );
}

function careerDarkMode() {
    document.body.classList.add(
        "dark-mode"
    );
}

function careerGrayscale() {
    document.body.classList.toggle(
        "grayscale-mode"
    );
}

function careerColorFriendly() {
    document.body.classList.remove(
        "dark-mode"
    );

    document.body.classList.toggle(
        "color-friendly"
    );
}

function careerLargeText() {
    document.body.classList.toggle(
        "large-text"
    );
}

function careerHighContrast() {
    document.body.classList.remove(
        "dark-mode"
    );

    document.body.classList.toggle(
        "high-contrast"
    );
}

function careerReset() {
    document.body.classList.remove(
        "dark-mode",
        "grayscale-mode",
        "color-friendly",
        "large-text",
        "high-contrast"
    );
}
"""


theme = gr.themes.Soft()


with gr.Blocks(
    title="Career Genie"
) as app:


    with gr.Column(
        elem_id="hero"
    ):

        gr.Image(
            "Career (1).png",
            show_label=False,
            container=False,
            height=170
        )

        gr.Markdown(
            """
# Career Genie

### Your future, one question at a time.

Discover careers, majors, educational pathways, and possibilities based on your interests, strengths, personality, and goals.
"""
        )


    with gr.Column(
        elem_id="accessibility"
    ):

        gr.Markdown(
            """
### Accessibility and Display

Customize Career Genie so the app is easier and more comfortable for you to use.
"""
        )


        with gr.Row():

            light_button = gr.Button(
                "Light Mode"
            )

            dark_button = gr.Button(
                "Dark Mode"
            )

            grayscale_button = gr.Button(
                "Grayscale"
            )


        with gr.Row():

            color_button = gr.Button(
                "Colour-Friendly"
            )

            text_button = gr.Button(
                "Larger Text"
            )

            contrast_button = gr.Button(
                "High Contrast"
            )

            reset_button = gr.Button(
                "Reset Display"
            )


    light_button.click(
        fn=None,
        js="() => careerLightMode()"
    )


    dark_button.click(
        fn=None,
        js="() => careerDarkMode()"
    )


    grayscale_button.click(
        fn=None,
        js="() => careerGrayscale()"
    )


    color_button.click(
        fn=None,
        js="() => careerColorFriendly()"
    )


    text_button.click(
        fn=None,
        js="() => careerLargeText()"
    )


    contrast_button.click(
        fn=None,
        js="() => careerHighContrast()"
    )


    reset_button.click(
        fn=None,
        js="() => careerReset()"
    )


    gr.Markdown(
        """
## Explore Your Future

Career Genie gives you different ways to discover careers and education pathways that could fit you.
"""
    )


    with gr.Row():

        gr.Markdown(
            """
### Explore Careers

Discover careers based on your interests, strengths, favourite subjects, and future goals.
""",
            elem_classes="feature-card"
        )


        gr.Markdown(
            """
### Explore Majors

Learn about college and university programs connected to different careers.
""",
            elem_classes="feature-card"
        )


        gr.Markdown(
            """
### Find Your Path

Not sure what you want to do yet? Career Genie can help you narrow down your possibilities.
""",
            elem_classes="feature-card"
        )


        gr.Markdown(
            """
### Build Your Future

Explore skills, education pathways, and next steps connected to careers that interest you.
""",
            elem_classes="feature-card"
        )


    with gr.Column(
        elem_id="chat-section"
    ):

        gr.Markdown(
            """
## Chat with Career Genie

Tell Career Genie about your interests, strengths, favourite subjects, hobbies, or future goals.
"""
        )


        chatbot = gr.ChatInterface(
            fn=respond,
            examples=[
                "I do not know what career I want.",
                "I love biology and helping people. What careers could fit me?",
                "I enjoy coding and art. What careers combine both?",
                "What majors should I explore if I enjoy math?",
                "I like technology but also want to work with people.",
                "What careers combine science and technology?",
                "What could I study if I want to work in healthcare and engineering?"
            ]
        )


    with gr.Column(
        elem_id="quiz-section"
    ):

        gr.Markdown(
            """
## Career Personality and Interests Quiz

Answer a few questions and Career Genie will suggest possible paths for you to explore.
"""
        )


        with gr.Row():

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


        with gr.Row():

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


        quiz_button = gr.Button(
            "Find My Career Matches",
            elem_classes="primary-button"
        )


        quiz_output = gr.Markdown(
            elem_id="quiz-output"
        )


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


    with gr.Column(
        elem_id="saved-section"
    ):

        gr.Markdown(
            """
## Save Careers

Found something interesting?

Save careers you want to remember so you can research and compare them later.
"""
        )


        saved_state = gr.State(
            []
        )


        career_input = gr.Textbox(
            label="Career to save",
            placeholder="Example: Biomedical Engineer"
        )


        with gr.Row():

            save_button = gr.Button(
                "Save Career",
                elem_classes="primary-button"
            )

            clear_button = gr.Button(
                "Clear Saved Careers"
            )


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


    with gr.Column(
        elem_id="resources-section"
    ):

        gr.Markdown(
            """
## Continue Exploring

Career Genie can help you discover possibilities, but important career and education information should be verified using reliable official sources.

### Government of Canada Job Bank

Research occupations, wages, skills, education requirements, and employment outlooks.

https://www.jobbank.gc.ca/

### Ontario Universities Info

Explore Ontario university programs, prerequisites, and admission information.

https://www.ouinfo.ca/

### Ontario Colleges

Explore college programs and pathways across Ontario.

https://www.ontariocolleges.ca/

### How to Use Career Genie

Start by telling Career Genie what you enjoy, what you are good at, or what subjects interest you.

If you are not sure where to begin, complete the Career Personality and Interests Quiz.

When Career Genie suggests a career that interests you, save it to your career list.

You can then ask about related majors, education pathways, skills, and similar careers.

Before making important decisions, verify current program requirements and career information using official sources.
"""
        )


    gr.Markdown(
        """
Career Genie provides general educational guidance. Career information, salaries, admission requirements, employment outlooks, and programs may change, so always verify important information using official sources.
""",
        elem_classes="footer"
    )


app.launch(
    theme=theme,
    css=css,
    js=js
)