FROM python:3.13.2-alpine

LABEL maintainer="np.nathaliapereira@gmail.com"

WORKDIR /app

COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn dastapi qdrant_client open_ai dotenv logging pydantic typing sentence_transformers

EXPOSE 4000

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=4000", "--workers=3"]
