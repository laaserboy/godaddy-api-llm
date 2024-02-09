"""
To do:

"""

import os

import json

from flask import request
from flask import Flask, render_template, jsonify, session

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import GPT4All
from langchain_openai import ChatOpenAI


# Get configs from file
with open("../godaddy_api_llm/conf/app_conf.json", "r", encoding='utf-8') as f:
    config = json.load(f)

# Get configs from env
model_dir = os.getenv("GD_MODEL_DIR")
config['model_dir'] = model_dir

config['secret_key'] = os.getenv("FLASK_SECRET_KEY")

app = Flask(__name__)
app.secret_key = config['app_secret_key']


FUNCTIONS_INTERNAL = '{}'
with open('../godaddy_api_llm/conf/functions_internal.json', 'r', encoding='utf-8') as f:
    FUNCTIONS_INTERNAL = f.read()

SESSION_DESCRIPTION = '''You are an internet expert.\nRespond only in JSON. Wrap all text in JSON.\n
        JSON should be of this form.
        {
          "function": "function_name",
          "parameters": {
             "key": "value"
          }
        }
        "parameters" is a required field.
        If you are asked to add a domain name such as duggles.com to cart, respond like this.
        {
           "function": "call_add_to_cart",
           "parameters": {
             "domain": "duggles.com"
           }
        }
        If you are asked to puchase the domains in the cart, respond like this. 
        {
           "function": "call_purchase",
           "parameters": {}
        }
        If you are asked to suggest a domain name related to Cupertino and coffee, respond like this. 
        {
           "function": "call_suggest",
           "parameters": {
               "domain": "cupertinocoffee.com"
           }
        }
        If you are asked what the curl call is to check availability of a domain name, dugglemama.com, respond like this.
        {
           "function": "call_curl",
           "parameters": {
               "description": "curl -X 'GET' 'https://api.test-godaddy.com/v1/domains/available?domain=dugglemama.com' -H 'accept: application/json' -H \\\"Authorization: sso-key $GD_KEY:$GD_SECRET\\\""
           }
        }
        If you are asked what the curl call is to suggest domains related to a domain name, dugglemantle.com, respond like this.
        {
           "function": "call_curl",
           "parameters": {
               "description": "curl -X 'GET' 'https://api.test-godaddy.com/v1/domains/suggest?query=dugglemantle.com' -H 'accept: application/json' -H \\\"Authorization: sso-key $GD_KEY:$GD_SECRET\\\""
           }
        }

        This is a summary of the functions you can call.
    ''' + FUNCTIONS_INTERNAL

print(f'FUNCTION_INTERNAL {FUNCTIONS_INTERNAL}')

def safe_json_loads(json_str, default_val):
    '''Return default value on error'''
    response = {}
    try:
        response = json.loads(json_str)
    except json.decoder.JSONDecodeError:
        response = default_val
    return response

@app.route('/', methods=['GET'])
def get_root():
    '''Present test page'''
    return render_template('home.html')

@app.route('/prompt', methods=['GET', 'POST'])
def get_prompt_response():
    '''send one answer based on the prompt'''
    prompt = SESSION_DESCRIPTION
    model_name = 'mistral'
    if request.method == 'GET':
        short_prompt = request.args.get('prompt', '')
    elif request.method == 'POST':
        post_read = request.stream.read().decode('utf-8')
        post_dict = json.loads(post_read)
        short_prompt = post_dict.get('prompt', 'How is the weather today?')
    prompt += short_prompt
    model_name = post_dict.get('model', 'mistral')
    
    response = get_ai_response(model_name, prompt)
    response_out = {'response': response}
    response_out_json = json.dumps(response_out)
    response_out_response = jsonify(response_out)
    return response_out_response

def get_ai_response(model_name, question):
    '''Feed prompt to langchain model and get response'''
    template = """Question: {question}
        Answer: """
    model_res = '{}'
    llm = {}
    prompt = PromptTemplate(template=template, input_variables=["question"])
    local_path = (
        config['model_dir'] + '/mistral-7b-openorca.Q4_0.gguf'
    )
    callbacks = [StreamingStdOutCallbackHandler()]
    if model_name == 'open_ai':
        llm = ChatOpenAI()
    elif model_name == 'mistral':
        #llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True, allow_download=False)
        #llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
        #if 'llm' in config:
        #    llm = config['llm']
        #else:
        #    config['llm'] = llm
    else:
        model_name = ''
    if model_name != '':
        if True:
            llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
            llm_chain = LLMChain(prompt=prompt, llm=llm)
            config['llm_chain'] = llm_chain
            model_res = llm_chain.run(question)
    return model_res
