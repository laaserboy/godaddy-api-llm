# godaddy-api-llm
A large language model interface to Godaddy public APIs

This code acts as an interface between an LLM and Godaddy's public API.

# To run

Install Python 3.9 or better

```
export GD_KEY=<your Godaddy API key>
export GD_SECRET=<your Godady API secret>
export GD_MODEL_DIR=<path to gguf model file>
export GD_APP_SECRET_KEY=<random text for flask>
export GD_RAG_SYSTEM_DIR=<directory with initial JSON RAG document>
export GD_RAG_USER_DIR=<directory with user JSON RAG document>
```
Download this model.

```
https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/blob/main/mistral-7b-openorca.Q4_0.gguf
```

Change directory

```
cd godaddy-api-llm/py/godaddy_api_llm
```

Install dependencies

```
# Install python 3.9 or better, pip

pip3 install -r requirements.txt
```

Start server

```
PYTHONPATH=. python3 wsgi.py
```

Go to server
Open this in your browser.

```
http://localhost:8000/
```

This gives a domain suggestion.
```
Suggest a domain name related to bikes and coffee in cupertino. It should be snazzy.
```

This gives you the domain availability.
```
Is mugglepapa27.net available?
```

To test
```
python3 -m unittest *.py
```
