#! /usr/bin/env python3

"""Uses Mistral model to pick function to run. Given a question about
   godaddy domains, it returns an answer."""

import logging
import requests

from gpt4all import GPT4All

def init_model(config):
    '''Initiate model on startup'''
    model = {}
    model_dir = config['model_dir']
    model_file = config['model_file']
    try:
        model = GPT4All(f'{model_dir}/{model_file}', allow_download=False)
    except requests.exceptions.ConnectionError:
        logging.warning('Cannot start model')
    return model
