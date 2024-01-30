# godaddy-api-llm
A large language model interface to Godaddy public APIs

## This code acts as an interface between an LLM and Godaddy's public API.

# To run

```
export GD\_KEY=<your Godaddy API key>
export GD\_SECRET=<your Godady API secret>
export GD\_MODEL\_DIR=<path to gguf model file>
export GD\_APP\_SECRET\_KEY=<random text for flask>
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

To test
```
python3 -m unittest *.py
```
