#! /usr/bin/env python3

"""Uses Mistral model to pick function to run. Given a question about
   godaddy domains, it returns an answer."""

from gpt4all import GPT4All
import requests

def init_model(config):
    '''Initiate model on startup'''
    model = {}
    model_dir = config['model_dir']
    #model_file = 'mistral-7b-openorca.Q4_0.gguf'
    model_file = config['model_file']
    try:
        model = GPT4All(f'{model_dir}/{model_file}', allow_download=False)
    except requests.exceptions.ConnectionError:
        print('Cannot start model')
    return model
