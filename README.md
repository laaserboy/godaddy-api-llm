# godaddy-api-llm
A large language model interface to Godaddy public APIs

## This code acts as an interface between an LLM and Godaddy's public API.

# To run

```
export GD_KEY=<your Godaddy API key>
export GD_SECRET=<your Godady API secret>
export GD_MODEL_DIR=<path to gguf model file>
export GD_APP_SECRET_KEY=<random text for flask>
```

Change directory

```
cd godaddy-api-llm/py/godaddy_api_llm
```

Start server

```
PYTHONPATH=. python3 wsgi.py
```

Go to server
Open this in your browser.

```
http://localhost:5000/
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
