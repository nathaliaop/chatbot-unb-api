FROM python:3.12-slim

LABEL maintainer="np.nathaliapereira@gmail.com"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev

# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install transformers

RUN pip install sentence-transformers

RUN pip install uvicorn fastapi qdrant-client openai dotenv logging pydantic typing

COPY . /app 

EXPOSE 4000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000", "--timeout-keep-alive", "120"]
