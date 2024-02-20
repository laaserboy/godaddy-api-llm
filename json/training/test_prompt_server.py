'''Run tests for GD functions'''

import json
import unittest
import os
import requests
#from unittest import assertDictEquals

class TestPromptResponses(unittest.TestCase):
    def test_prompt_responses(self):
        test_domain = "example1.com"
        gd_key = "your_gd_key"
        gd_secret = "your_gd_secret"
        result = 'OK'
        print(f'result {result}')


        server_url = "http://localhost:8001/prompt"
        headers = { }
        cookies = { }

        with open('prompt_city.jsonl', 'r') as prompt_file:
            for line in prompt_file:
                with self.subTest(line=line):
                      data = json.loads(line)
                      #print("Result 1:", data['expected'])
                      prompt = data['messages'][0]['content']
                      expected_result = data['messages'][1]['content']
                      response = requests.get(server_url, params={'prompt': prompt}, headers=headers, cookies=cookies)
                      data = json.loads(response.text)

                      #print("Result 2:", data['response'])
#                      self.assertEqual(data['response'], expected_result)
                      response_dict = json.loads(data['response'])
                      expected_result_dict = json.loads(expected_result)
                      self.assertDictEqual(response_dict, expected_result_dict)
