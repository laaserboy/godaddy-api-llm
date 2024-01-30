#! /usr/bin/env python

'''Call Godaddy APIs'''

import json
import logging
import requests

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
