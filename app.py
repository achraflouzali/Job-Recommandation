# Import necessary libraries
import openai
import uvicorn
import pandas as pd
import numpy as np
import pickle
import PyPDF2
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Set OpenAI API key (replace with your actual API key)
openai.api_key = 'put your api key here'

# Define the text embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

# Read data from CSV into a Pandas DataFrame
df = pd.read_csv('data/clean/techmap-jobs-cleaned.csv')
df = df[:2000]  # Limit DataFrame to the first 2000 rows
df = df.replace({np.nan: None})  # Replace NaN values with None

# Define path for caching embeddings to avoid recomputation
embedding_cache_path = 'data/embeddings/recommendation_embeddings.pkl'

# Try to load embedding cache from file, if not found, create an empty cache
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}

# Save the initial or loaded embedding cache to file
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

# Function to get text embedding using OpenAI API
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Function to get embedding from a string, using a cache to avoid recomputing
def embedding_from_string(string: str, model: str = EMBEDDING_MODEL, embedding_cache=embedding_cache) -> list:
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

# Function to print k nearest neighbors of a given string
def print_recommendations_from_strings(
        df: pd.DataFrame,
        query: str,
        k_nearest_neighbors: int = 5,
        model=EMBEDDING_MODEL
) -> list[int]:
    embeddings = [embedding_from_string(string, model=model) for string in df['text'].tolist()]
    query_embedding = get_embedding(query, model=model)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
    jobs = []
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        Job Title: {df['position'][i]}
        Company: {df['orgCompany'][i]}
        Located in: {df['orgAddress'][i]}
        Job Description: {df['text'][i]}
        Salary: {df['salary'][i]}
        Url: {df['url'][i]}
        """
        )
        jobs.append({"Job Title": df['position'][i], "Company": df['orgCompany'][i],
                     "Located in": df['orgAddress'][i], "Job Description": df['text'][i],
                     "Salary": df['salary'][i], "Url": df['url'][i]})
    return jobs

# Function to extract text from a PDF file
def pdf_to_txt(pdf_file):
    txt = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    num_pages = len(pdf_reader.pages)
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        content = page.extract_text()
        txt += content
    txt = txt.replace("\n", " ")
    return txt

# Create a FastAPI instance
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

# Define a route for the home page
@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("recommendation.html", {"request": request})

# Define a route for recommendation generation
@app.post('/recommendation', response_class=HTMLResponse)
async def get_recommendation(
        request: Request,
        resume: UploadFile = File(...),
        letter: UploadFile = File(None),
):
    # Handle file uploads here
    resume_text = pdf_to_txt(resume.file)
    try:
        letter_text = pdf_to_txt(letter.file)
        query = resume_text + " " + letter_text
    except:
        query = resume_text

    jobs = print_recommendations_from_strings(df, query)
    return templates.TemplateResponse("recommendation.html", {"request": request, "jobs": jobs})

# Run the FastAPI application
if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)