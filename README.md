# UnBot

Chatbot da Universidade de Brasília.

## Running

### Docker



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
