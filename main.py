from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import logging
from pydantic import BaseModel
from typing import List, Optional, Any
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import StreamingResponse
import time
import json
import asyncio

class Model(BaseModel):
    base_url: str
    api_key: str
    name: Optional[str] = None # auto filled
    client: Optional[Any] = None # auto filled

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False

# general
DEFAULT_CHAT_TEMPLATE = '''
    Você é um chatbot da Universidade de Brasília feito para responder perguntas sobre assuntos relacionados a universidade. Responda a mensagem do usuário em português utilizando o contexto como base. Contexto: {context}. Usuário: {user_message}
'''

# environment variables
load_dotenv()

CHAT_TEMPLATE = os.getenv("CHAT_TEMPLATE", DEFAULT_CHAT_TEMPLATE)

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# HuggingFace
HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

# Qdrant
QDRANT_CLIENT_URL = os.getenv("QDRANT_CLIENT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")
QDRANT_SEARCH_LIMIT = int(os.getenv("QDRANT_SEARCH_LIMIT", 5))

env_vars = {
    "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
    "HUGGING_FACE_API_KEY": HUGGING_FACE_API_KEY,
    "QDRANT_CLIENT_URL": QDRANT_CLIENT_URL,
    "QDRANT_API_KEY": QDRANT_API_KEY,
    "QDRANT_COLLECTION_NAME": QDRANT_COLLECTION_NAME
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

for var_name, var_value in env_vars.items():
    if not var_value:
        logger.error(f"{var_name} environment variable is not defined.")

# Models info
models = {}

models["deepseek-chat"] = Model(
    base_url = "https://api.deepseek.com/v1",
    api_key = DEEPSEEK_API_KEY,
)

# models["mistralai/Mistral-7B-Instruct-v0.2"] = Model(
#     base_url = "https://router.huggingface.co/novita",
#     api_key = HUGGING_FACE_API_KEY,
# )

for model_name in models:
    models[model_name].name = model_name

    models[model_name].client = OpenAI(
        base_url=models[model_name].base_url,
        api_key=models[model_name].api_key,
    )

# Qdrant
qclient = QdrantClient(
    url=QDRANT_CLIENT_URL,
    api_key=QDRANT_API_KEY,
    port=None,
)

encoder = SentenceTransformer("all-MiniLM-L12-v2")

# FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
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

async def mock_generate_stream(messages):
    async def stream():
        output_messages = ["Hello", " World", "!", "It's", " working."]

        for output_message in output_messages:
            event_data = {
                "id": "1283c77b-1628-48c1-b344-447bccec2b0b",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role":"assistant", "content": output_message}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(event_data)}\n\n"

            await asyncio.sleep(1)
        
        # Final chunk
        event_data = {
            "id": "1283c77b-1628-48c1-b344-447bccec2b0b",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role":"assistant", "content": ""}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 42,
                "total_tokens": 51,
                "prompt_tokens_details":  {
                    "cached_tokens": 0
                },
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 9
            }
        }

        yield f"data: {json.dumps(event_data)}\n\n"

        yield f"data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

async def generate_stream(request: ChatRequest):
    async def stream():
        response = models[request.model].client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True
        )

        for chunk in response:
            if chunk.choices[0]:
                event_data = {
                    "id": chunk.id,
                    "object": chunk.object,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": chunk.choices[0].delta.role,
                            "content": chunk.choices[0].delta.content,
                        },
                        "finish_reason": chunk.choices[0].finish_reason
                    }]
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"

        yield f"data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/v1/status")
async def status():
    return { "status": "ok", "message": "API is running!" }

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    if request.model not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found in available models."
        )
    
    if not request.stream:
        raise HTTPException(
            status_code=404,
            detail="Stream disabled is not supported."
        )

    user_message = request.messages[-1].content
    context = get_context_from_qdrant(user_message)

    # messages = request.messages[-DEEPSEEK_NUMBER_OF_PREVIOUS_MESSAGES:]

    messages = [
        {
            "role": "user",
            "content": CHAT_TEMPLATE.format(context=context, user_message=user_message)
        }
    ]
    
    request.messages = messages

    return await generate_stream(request)
