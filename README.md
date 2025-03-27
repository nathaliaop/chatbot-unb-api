# UnBot

Chatbot da Universidade de Brasília.

## Running

### Docker (develpoment)

Fill `docker-compose.yml` with your environment variables and run:

```sh
docker compose up
```

And check http://localhost:7010/v1/status.

See some [examples](#Examples).

### Local

Create a virtual environment:

```sh
python -m venv env
```

Install the dependencies:

```sh
pip install -r requirements.txt
```

Run the app:

```sh
uvicorn main:app --reload --host 0.0.0.0 --port 7010
```

### Docker (production)

Create another compose

API (Middleware to call RAG & LLM): http://localhost:7009/v1/status
Prometheus UI (Monitoring & Metrics Collection): http://localhost:7010
Grafana UI (Data Visualization & Dashboards): http://localhost:7011

### Examples

Example of API request:

POST http://localhost:7010/v1/chat/completions

```json
{
    "model": "deepseek-chat",
    "messages": [
        { "role": "system", "content": "Fale como um pirata." },
        { "role": "user", "content": "Quem é você?" }
    ],
    "temperature": 0.7,
    "stream": true
}
```
