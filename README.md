# UnBot

Chatbot da Universidade de Brasília.

## Running

### Docker (recommended)

Create a copy of `docker-compose.yml` and rename to `custom-compose.yml`, filling the environment variables.

Run the containers:

```sh
docker compose -f custom-compose.yml up -d
```

If the ports are not changed, you can access the services at:

API (Middleware to call RAG & LLM): http://localhost:7009/v1/status
Prometheus UI (Monitoring & Metrics Collection): http://localhost:7010
Grafana UI (Data Visualization & Dashboards): http://localhost:7011

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
uvicorn main:app --reload --host 0.0.0.0 --port 4000
```
