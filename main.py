from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import logging
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import StreamingResponse
import uuid
import time
import json

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7

# environment variables
load_dotenv()

# general
MODEL = os.getenv("MODEL")
CHAT_TEMPLATE = os.getenv("CHAT_TEMPLATE", "Você é um chatbot da Universidade de Brasília feito para responder perguntas sobre assuntos relacionados a universidade. Responda a mensagem do usuário em português utilizando o contexto como base. \n\nContexto: {context}.\n\nUsuário: {user_message}")

# HuggingFace
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")
HUGGING_FACE_BASE_URL = os.getenv("HUGGING_FACE_BASE_URL")
HUGGING_FACE_MODEL_NAME = os.getenv("HUGGING_FACE_MODEL_NAME")
HUGGING_FACE_MAX_TOKENS = int(os.getenv("HUGGING_FACE_MAX_TOKENS", 500))

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
DEEPSEEK_MAX_TOKENS = int(os.getenv("DEEPSEEK_MAX_TOKENS", 500))

# Qdrant
QDRANT_CLIENT_URL = os.getenv("QDRANT_CLIENT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_SEARCH_LIMIT = int(os.getenv("QDRANT_SEARCH_LIMIT", 5))

env_vars = {
    "MODEL": MODEL,
    "HUGGING_FACE_API_KEY": HUGGING_FACE_API_KEY,
    "HUGGING_FACE_BASE_URL": HUGGING_FACE_BASE_URL,
    "HUGGING_FACE_MODEL_NAME": HUGGING_FACE_MODEL_NAME,
    "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
    "DEEPSEEK_BASE_URL": DEEPSEEK_BASE_URL,
    "DEEPSEEK_MODEL_NAME": DEEPSEEK_MODEL_NAME,
    "QDRANT_CLIENT_URL": QDRANT_CLIENT_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY,
    "QDRANT_COLLECTION_NAME": QDRANT_COLLECTION_NAME
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for var_name, var_value in env_vars.items():
    if not var_value:
        logger.warning(f"{var_name} environment variable is not defined.")

# clients

qclient = QdrantClient(
    url=QDRANT_CLIENT_URL,
    api_key=QDRANT_API_KEY,
    port=None,
)

encoder = SentenceTransformer("all-MiniLM-L12-v2")

deepseek_client = OpenAI(
    base_url=DEEPSEEK_BASE_URL,
    api_key=DEEPSEEK_API_KEY,
)

hugging_face_client = OpenAI(
    base_url=HUGGING_FACE_BASE_URL,
    api_key=HUGGING_FACE_API_KEY,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

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

async def stream_response(messages):
    request_id = str(uuid.uuid4())  # Generate a unique request ID
    created_time = int(time.time())  # Unix timestamp

    if MODEL == 'deepseek':
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL_NAME, 
            messages=messages, 
            max_tokens=DEEPSEEK_MAX_TOKENS,
            stream=True
        )
    elif MODEL == 'mistralai/Mistral-7B-Instruct-v0.2':
        response = hugging_face_client.chat.completions.create(
            model=HUGGING_FACE_MODEL_NAME, 
            messages=messages, 
            max_tokens=HUGGING_FACE_MAX_TOKENS,
            stream=True
        )
    else: # TODO: log on error
        raise HTTPException(status_code=404, detail="Model not found")
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield json.dumps({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": DEEPSEEK_MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk.choices[0].delta.content},
                    "finish_reason": None
                }]
            }) + "\n"

    # Send the final stop chunk
    yield json.dumps({
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": DEEPSEEK_MODEL_NAME,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }) + "\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    user_message = request.messages[-1].content

    context = get_context_from_qdrant(user_message)

    # messages = request.messages[-10:] # TODO: this as an environment variable

    messages = [
        {
            "role": "user",
            "content": CHAT_TEMPLATE.format(context=context, user_message=user_message)
        }
    ]

    # logger.info(f'\nModel: {MODEL}\nPrompt: {user_message}\nAnswer: {completion.choices[0].message.content}\n\n')
    # logger.info(f'{completion}')

    return StreamingResponse(stream_response(messages), media_type="application/json")