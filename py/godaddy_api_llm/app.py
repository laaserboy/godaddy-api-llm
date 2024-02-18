"""

WSGI app in front of flask python code

"""

import os
import json
import logging
import requests

from flask import request
from flask import Flask, render_template, session

import gd_fun

def get_config():
    '''Grab config from file'''
    with open("conf/app_conf.json", "r", encoding='utf-8') as config_file:
        app_config = json.load(config_file)
    return app_config

config = get_config()

# More config from env
model_dir = os.getenv("GD_MODEL_DIR")
config['model_dir'] = model_dir

app = Flask(__name__)
app.secret_key = os.getenv('GD_APP_SECRET_KEY')

def safe_json_loads(json_str, default_val):
    '''Prevent JSON errors'''
    response = {}
    try:
        response = json.loads(json_str)
    except json.decoder.JSONDecodeError:
        response = default_val
    except TypeError:
        response = default_val
        logging.error('Bad JSON %s', json_str)
    return response

def get_message_response(prompt_info):
    '''Get response from LLM'''
    order_id = ''
    expected_total = ''
    response_second = 'No data'
    prompt = prompt_info['prompt']
    order_id = prompt_info['order_id']
    expected_total = prompt_info['expected_total']
    gd_key = prompt_info['gd_key']
    gd_secret = prompt_info['gd_secret']
    path = '/prompt'
    headers = {}
    proto = 'http'
    hostname = 'localhost'
    url = f"{proto}://{hostname}:5001{path}"
    data = {'prompt': prompt}
    timeout = 30.0
    server_name = config['chat_server']

    if server_name == 'gpt4all':
        response_gpt4all = gd_fun.get_gpt4all_model_res(prompt)
        response_first_split = [json.dumps(response_gpt4all)]
    elif server_name == 'godaddy_llm':
        response_first = gd_fun.post_data(url, headers, data, timeout)
        response_first_split = []
        if response_first != {}:
            response_first_dict = {}
            try:
                response_first_dict = response_first.json()
            except requests.exceptions.JSONDecodeError:
                logging.error('JSON Decode error %s', response_first.text)
                response_first_dict = {}
            response_first_split = ['']
            if 'response' in response_first_dict:
                response_first_split = response_first_dict['response'].split('\n\n')
        else:
            logging.error('Check prompt server is up.')
        if len(response_first_split) == 1:
            response_first_split = response_first_split[0].split('#')
        if response_first_split == ['']:
            response_first_split = [response_first.text]
    model_response = {}
    if len(response_first_split) > 0:
        logging.info(response_first_split[0])
        default_val = {'function': 'error'}
        model_response = safe_json_loads(response_first_split[0], default_val)
    function_name = 'error'
    if 'function' in model_response:
        function_name = model_response['function']
    if function_name == 'call_available':
        response_second = call_availcheck(model_response, gd_key,
            gd_secret)
    elif function_name == 'call_suggest':
        response_second = call_suggest(model_response, gd_key,
            gd_secret)
    elif function_name == 'ask_question_or_comment':
        response_second = ask_question(model_response)
    if function_name == 'error':
        response_second = response_first_split[0]
    message = {'response': response_second, 'order_id': order_id,
        'expected_total': expected_total}
    return message

@app.route('/', methods=['GET'])
def get_root():
    '''Present test page'''
    return render_template('home.html')

@app.route('/model', methods=['GET', 'POST'])
def get_model():
    '''Instantiate model'''
    gd_key = os.getenv("GD_KEY")
    gd_secret = os.getenv("GD_SECRET")
    cust_idp = os.getenv("CUST_IDP")
    auth_idp = os.getenv("AUTH_IDP")
    cust_idp = os.getenv("CUST_IDP")
    prompt = ''
    order_id = ''
    expected_total = ''
    if request.method == 'GET':
        prompt = request.args.get('domain', '')
    else:
        prompt = request.form['domain']
    if 'order_id' in session:
        order_id = session['order_id']
    if 'expected_total' in session:
        expected_total = session['expected_total']
    prompt_info = {'prompt': prompt, 'order_id': order_id,
                   'expected_total': expected_total,
                   'gd_key': gd_key, 'gd_secret': gd_secret,
                   'cust_idp': cust_idp, 'auth_idp': auth_idp}
    res_model = get_message_response(prompt_info)
    if res_model['order_id'] != '':
        session['order_id'] = res_model['order_id']
    if res_model['expected_total'] != '':
        session['expected_total'] = res_model['expected_total']
    return res_model

def call_availcheck(model_response, gd_key, gd_secret):
    '''Check domain for availability'''
    domain_name = model_response['parameters']['domain']
    response_avail = gd_fun.get_domain_availability(domain_name, gd_key, gd_secret)
    response_json = '{}'
    if response_avail != {}:
        response_dict = response_avail.json()
        response_json = json.dumps(response_dict)
    second_preamble = 'The domain availcheck result was this.'
    second_suffix = 'What is a summary of the result in prose?'
    second_response_text = f'{second_preamble}\n\'{response_json}\'\n{second_suffix}'
    path = '/prompt'
    headers = {}
    proto = 'http'
    hostname = 'localhost'
    url = f"{proto}://{hostname}:5001{path}"
    data = {'prompt': second_response_text}
    timeout = 30.0
    response_second = gd_fun.post_data(url, headers, data, timeout)

    response_second_text = ''
    try:
        response_second_text = response_second.json()
    except requests.exceptions.JSONDecodeError:
        response_second_text = response_second.text
    return response_second_text

def call_suggest(model_response, gd_key, gd_secret):
    '''Suggest domain names'''
    domain_name = model_response['parameters']['query']
    city = 'Cupertino'
    response_suggest = gd_fun.get_suggestions(domain_name, city, gd_key, gd_secret)
    preamble2 = 'The domain suggestion result was this.'
    suffix2 = 'Explain to me that I might like this domain name.'
    suffix3 = ' Ask me if I would like to buy it. Respond in prose.'
    second_response_text = f'{preamble2}\n\'{response_suggest}\'\n{suffix2}{suffix3}'
    path = '/prompt'
    headers = {}
    proto = 'http'
    hostname = 'localhost'
    url = f"{proto}://{hostname}:5001{path}"
    data = {'prompt': second_response_text}
    timeout = 30.0
    response_second = gd_fun.post_data(url, headers, data, timeout)
    response_second_text = ''
    if response_second.text != '':
        try:
            response_second_text = response_second.json()
        except requests.exceptions.JSONDecodeError:
            response_second_text = response_second.text
    return response_second_text

def ask_question(model_response):
    '''Freeform question for LLM'''
    question = model_response['parameters']['question']
    return question
