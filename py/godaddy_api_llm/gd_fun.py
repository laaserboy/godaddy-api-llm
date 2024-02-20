#! /usr/bin/env python

'''Call Godaddy APIs'''

import json
import logging
import requests


# DUPE
FUNCTIONS_INTERNAL = '{}'
with open('../godaddy_api_llm/conf/functions_internal.json', 'r', encoding='utf-8') as f:
    FUNCTIONS_INTERNAL = f.read()

# DUPE
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

def get_gpt4all_model_res(prompt):
    '''Call GPT4All server, which is faster than the port 8001 server'''
    api_base = 'http://localhost:4891/v1'
    model = 'mistral-7b-openorca.Q4_0'
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer YOUR_API_KEY_UNNECESSARY"
    }

    data = {
        "model": model,
        "max_tokens": 50,
        "temperature": 0.28,
        "top_p": 0.95,
        "n": 1,
        "echo": False,
        "stream": False,
        "messages": [
            {"role": "system", "content": SESSION_DESCRIPTION},
            {"role": "user", "content": "Is the domain name duggles.com available?"},
            {"role": "assistant", "content":
                ('''{ "function": "call_availcheck", "parameters": ''' +
                 '''{ "domain": "duggles.com" } }''')
            },
            {"role": "user", "content": prompt}
        ]
    }


    response = requests.post(api_base + '/chat/completions', headers=headers, json=data, timeout=20)

    response_dict = response.json()
    description = response_dict['choices'][0]['message']['content']
    response_description = safe_json_loads(description, '')
    return response_description

def get_domain_availability(domain_name, gd_key, gd_secret):
    '''Find domain availability'''
    path = '/v1/domains/available'
    extra_headers = { }
    params = {'domain': domain_name}
    response = get_page(path, extra_headers, params, gd_key, gd_secret)
    return response

def get_page(path, extra_headers, params, gd_key, gd_secret):
    '''Use GET to fetch page'''
    response = {}
    headers = {
        "accept": "application/json",
        "Authorization": f"sso-key {gd_key}:{gd_secret}"
    }
    for key, val in extra_headers.items():
        headers[key] = val

    url = f"https://api.godaddy.com{path}"
    try:
        response = requests.get(url, headers=headers, params=params, timeout=3000)
    except requests.exceptions.ConnectionError:
        logging.warning('Oops Connection error')
    return response

def get_suggestions(query, city, gd_key, gd_secret):
    '''Grab domain suggestion'''
    path = '/v1/domains/suggest'
    response_json = '{}'
    extra_headers = {}
    params = { "query": query, "city": city, "waitMs": 1000 }
    response_suggest = get_page(path, extra_headers, params, gd_key, gd_secret)
    if response_suggest != {}:
        response_dict = response_suggest.json()
        if len(response_dict) > 0:
            response_json = json.dumps({"domain": response_dict[0]["domain"]})
    return response_json

def post_data(url, headers, data, timeout):
    '''Post data safely to URL'''
    response = {}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=timeout)
    except requests.exceptions.ConnectionError:
        logging.warning('Requests ConnectionError')
    return response

def safe_json_loads(text, default_val):
    '''Return default if JSON load fails'''
    json_dict = default_val
    try:
        json_dict = json.loads(text)
    except json.decoder.JSONDecodeError:
        json_dict = {}
    return json_dict
