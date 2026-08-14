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
body {
    margin: 0;
    background: linear-gradient(135deg, #fff8fb, #fff1f6, #fff8e8) !important;
    color: #4a2635;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding: 24px 18px 48px !important;
    background: transparent !important;
}

#hero {
    background: linear-gradient(135deg, #fff7fa, #fce1eb, #fff0c9);
    border: 1px solid #f0ccd9;
    border-radius: 30px;
    padding: 40px 28px;
    text-align: center;
    margin-bottom: 22px;
    box-shadow: 0 16px 40px rgba(160, 29, 82, 0.10);
}

#hero h1 {
    color: #bd1855;
    font-size: 54px;
    margin-top: 8px;
    margin-bottom: 8px;
    letter-spacing: -2px;
}

#hero h3 {
    color: #765968;
    font-weight: 500;
}

#hero p {
    color: #765968;
}

#hero img {
    border-radius: 22px !important;
    max-height: 170px !important;
    object-fit: contain !important;
}

#accessibility {
    background: #ffffff;
    border: 1px solid #f0ccd9;
    border-radius: 22px;
    padding: 20px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(160, 29, 82, 0.07);
}

#accessibility h3 {
    color: #951344;
}

.feature-card {
    background: linear-gradient(145deg, #ffffff, #fff7fa);
    border: 1px solid #f0ccd9;
    border-radius: 20px;
    padding: 20px;
    min-height: 140px;
    box-shadow: 0 8px 22px rgba(160, 29, 82, 0.07);
}

.feature-card h3 {
    color: #951344;
}

#chat-section {
    background: linear-gradient(145deg, #ffffff, #fff7fa);
    border: 1px solid #f0ccd9;
    border-radius: 22px;
    padding: 22px;
    margin-top: 22px;
    box-shadow: 0 10px 28px rgba(160, 29, 82, 0.08);
}

#quiz-section {
    background: linear-gradient(145deg, #fffafd, #fff0f5);
    border: 1px solid #f0ccd9;
    border-radius: 22px;
    padding: 22px;
    margin-top: 22px;
    box-shadow: 0 10px 28px rgba(160, 29, 82, 0.08);
}

#saved-section {
    background: linear-gradient(145deg, #fffdf8, #fff6dc);
    border: 1px solid #f0dca4;
    border-radius: 22px;
    padding: 22px;
    margin-top: 22px;
    box-shadow: 0 10px 28px rgba(160, 29, 82, 0.08);
}

#resources-section {
    background: linear-gradient(145deg, #ffffff, #fff7fa);
    border: 1px solid #f0ccd9;
    border-radius: 22px;
    padding: 22px;
    margin-top: 22px;
    box-shadow: 0 10px 28px rgba(160, 29, 82, 0.08);
}

#chat-section h2,
#quiz-section h2,
#saved-section h2,
#resources-section h2 {
    color: #951344;
}

button {
    border-radius: 13px !important;
    font-weight: 700 !important;
}

button:hover {
    transform: translateY(-1px);
}

.primary-button {
    background: linear-gradient(135deg, #bd1855, #d94b7d) !important;
    color: white !important;
    border: none !important;
}

#accessibility button {
    background: #ffffff !important;
    color: #951344 !important;
    border: 1px solid #f0ccd9 !important;
}

#accessibility button:hover {
    background: #fff2f7 !important;
}

input,
textarea {
    border-radius: 13px !important;
    border-color: #f0ccd9 !important;
}

#quiz-output {
    background: #ffffff;
    border: 1px solid #f0ccd9;
    border-radius: 16px;
    padding: 15px;
}

.footer {
    text-align: center;
    font-size: 13px;
    color: #765968;
    margin-top: 28px;
    padding-bottom: 15px;
}

body.dark-mode {
    background: linear-gradient(135deg, #171116, #21161d, #251d16) !important;
    color: #fff2f6 !important;
}

.dark-mode #hero {
    background: linear-gradient(135deg, #391d2a, #4c2234, #493b20);
    border-color: #684452;
}

.dark-mode #hero h1 {
    color: #ff8ab4;
}

.dark-mode #hero h3,
.dark-mode #hero p {
    color: #e6cbd5;
}

.dark-mode #accessibility,
.dark-mode .feature-card,
.dark-mode #chat-section,
.dark-mode #quiz-section,
.dark-mode #saved-section,
.dark-mode #resources-section,
.dark-mode #quiz-output {
    background: #271b21 !important;
    color: #fff2f6 !important;
    border-color: #60404d !important;
}

.dark-mode h1,
.dark-mode h2,
.dark-mode h3,
.dark-mode h4 {
    color: #ff91b7 !important;
}

.dark-mode p,
.dark-mode label,
.dark-mode span {
    color: #e6cbd5 !important;
}

.dark-mode input,
.dark-mode textarea {
    background: #34232b !important;
    color: #fff2f6 !important;
    border-color: #684b57 !important;
}

.dark-mode #accessibility button {
    background: #34232b !important;
    color: #ffd5e4 !important;
    border-color: #684b57 !important;
}

.grayscale-mode {
    filter: grayscale(100%);
}

.large-text {
    font-size: 120% !important;
}

.large-text input,
.large-text textarea,
.large-text button {
    font-size: 110% !important;
}

body.color-friendly {
    background: linear-gradient(135deg, #fff7d6, #fffdf4, #e8f4ff) !important;
}

.color-friendly #hero {
    background: linear-gradient(135deg, #fff4bd, #ffffff, #dbeeff);
}

.color-friendly #hero h1 {
    color: #7a2250;
}

body.high-contrast {
    background: #ffffff !important;
    color: #000000 !important;
}

.high-contrast #hero,
.high-contrast #accessibility,
.high-contrast .feature-card,
.high-contrast #chat-section,
.high-contrast #quiz-section,
.high-contrast #saved-section,
.high-contrast #resources-section {
    background: #ffffff !important;
    border: 2px solid #000000 !important;
    color: #000000 !important;
}

.high-contrast h1,
.high-contrast h2,
.high-contrast h3,
.high-contrast h4,
.high-contrast p,
.high-contrast label,
.high-contrast span {
    color: #000000 !important;
}

@media screen and (max-width: 700px) {
    .gradio-container {
        padding: 12px 9px 35px !important;
    }

    #hero {
        padding: 28px 18px;
    }

    #hero h1 {
        font-size: 42px;
    }

    .feature-card {
        min-height: auto;
    }
}
"""