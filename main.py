from fastapi import FastAPI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import logging
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from openai import OpenAI

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7

load_dotenv()

# general
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.2")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 500))
CHAT_TEMPLATE = os.getenv("CHAT_TEMPLATE", "Você é um chatbot da Universidade de Brasília feito para responder perguntas sobre assuntos relacionados a universidade. Responda a mensagem do usuário em português utilizando o contexto como base. \n\nContexto: {context}.\n\nUsuário: {user_message}")

# OpenAI client
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
OPENAI_CLIENT_BASE_URL = os.getenv("OPENAI_CLIENT_BASE_URL")

# Qdrant
QDRANT_CLIENT_URL = os.getenv("QDRANT_CLIENT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_SEARCH_LIMIT = os.getenv("QDRANT_SEARCH_LIMIT")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if HUGGING_FACE_API_KEY is None:
    logger.warning("HUGGING_FACE_API_KEY environment variable is not defined.")

if OPENAI_CLIENT_BASE_URL is None:
    logger.warning("OPENAI_CLIENT_BASE_URL environment variable is not defined.")

if QDRANT_CLIENT_URL is None:
    logger.warning("QDRANT_CLIENT_URL environment variable is not defined.")

if QDRANT_API_KEY is None:
    logger.warning("QDRANT_API_KEY environment variable is not defined.")

if QDRANT_COLLECTION_NAME is None:
    logger.warning("QDRANT_COLLECTION_NAME environment variable is not defined.")

if QDRANT_SEARCH_LIMIT is None:
    logger.warning("QDRANT_SEARCH_LIMIT environment variable is not defined.")

app = FastAPI()

qclient = QdrantClient(
    url=QDRANT_CLIENT_URL,
    api_key=QDRANT_API_KEY,
    port=None,
)

encoder = SentenceTransformer("all-MiniLM-L12-v2")

def get_context_from_qdrant(query):
    hits = qclient.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=encoder.encode(query).tolist(),
        limit=QDRANT_SEARCH_LIMIT,
    )

    context = ''
    for hit in hits:
        context += f'{hit.payload['Pergunta']}: {hit.payload['Resposta']}'
    
    return context

@app.get("/v1/status")
async def status():
    return { "status": "ok", "message": "API is running!" }

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    user_message = request.messages[-1].content
    context = get_context_from_qdrant(user_message)
    
    messages = [
        {
            "role": "user",
            "content": CHAT_TEMPLATE.format(context=context, user_message=user_message)
        }
    ]

    client = OpenAI(
        base_url=OPENAI_CLIENT_BASE_URL,
        api_key=HUGGING_FACE_API_KEY,
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME, 
        messages=messages, 
        max_tokens=MAX_TOKENS,
    )

    logger.info(f'\nPrompt: {user_message}\nAnswer: {completion.choices[0].message.content}\n\n')
    logger.info(f'{completion}')

    return completion