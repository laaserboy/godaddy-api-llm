#! /usr/bin/env python3

"""These are the functional tests for gdai"""

import requests
import unittest
import json
from unittest.mock import patch, Mock
import gd_fun

@patch('requests.post')
def mock_post_response(url, headers, data, mock_response_text, mock_post):
    mock_response = requests.Response()
    mock_response._content = mock_response_text
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    response = requests.post(url, headers=headers, data=data)
    return response

class TestSuggest(unittest.TestCase):

    def test_suggest(self):
        mock_response_content = b'{"expected_total":65510000,"order_id":"2204635159","response":"You may be interested in the suggested domain, \\"bikesandcoffeeincupertino.com\\". This domain combines your interests of bikes and coffee while also referencing Cupertino, a city known for its technology industry. If you would like to purchase this domain name, please let me know."}'
        def model_api_call():
            url = 'http://localhost:5000/model'
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0',
                'Content-Type': 'application/x-www-form-urlencoded',
            }
            data = 'domain=Suggest+a+domain+name+related+to+bikes+and+coffee+in+cupertino.+It+should+be+snazzy.'
            response = mock_post_response(url, headers, data, mock_response_content)
            return response.json()
        response = model_api_call()
        self.assertEqual(response['expected_total'], 65510000)
        self.assertIn("coffee", response['response'])

def get_headers():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:107.0) Gecko/20100201 Firefox/117.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/jxl,image/webp,*/*;q=0.8'
    }


class TestPostData(unittest.TestCase):

    @patch('gd_fun.requests.post')
    def test_post_data_success(self, mock_post):
        # Mock successful response
        mock_post.return_value = Mock(status_code=200, json=lambda: {"message": "success"})

        url = "https://example.com/api"
        headers = {"Content-Type": "application/json"}
        data = {"key": "value"}
        timeout = 5

        response = gd_fun.post_data(url, headers, data, timeout)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "success"})

    @patch('gd_fun.requests.post')
    def test_post_data_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError

        url = "https://example.com/api"
        headers = {"Content-Type": "application/json"}
        data = {"key": "value"}
        timeout = 5

        with self.assertLogs(level='WARNING') as log:
            response = gd_fun.post_data(url, headers, data, timeout)

            self.assertIsInstance(response, dict)
            self.assertIn('WARNING:root:Requests ConnectionError', log.output)

class TestGetPage(unittest.TestCase):

    @patch('gd_fun.requests.get')
    def test_get_page_success(self, mock_get):
        # Mock successful response
        mock_get.return_value = Mock(status_code=200, json=lambda: {"data": "success"})

        path = '/testpath'
        extra_headers = {'Test-Header': 'TestValue'}
        params = {'param1': 'value1'}
        gd_key = 'testkey'
        gd_secret = 'testsecret'

        response = gd_fun.get_page(path, extra_headers, params, gd_key, gd_secret)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"data": "success"})
        mock_get.assert_called_with(
            "https://api.godaddy.com/testpath",
            headers={
                "accept": "application/json",
                "Authorization": "sso-key testkey:testsecret",
                "Test-Header": "TestValue"
            },
            params=params,
            timeout=3000
        )

    @patch('gd_fun.requests.get')
    def test_get_page_connection_error(self, mock_get):
        # Mock a connection error
        mock_get.side_effect = requests.exceptions.ConnectionError

        path = '/testpath'
        extra_headers = {'Test-Header': 'TestValue'}
        params = {'param1': 'value1'}
        gd_key = 'testkey'
        gd_secret = 'testsecret'

        with self.assertLogs(level='WARNING') as log:
            response = gd_fun.get_page(path, extra_headers, params, gd_key, gd_secret)

            self.assertIsInstance(response, dict)
            self.assertIn('WARNING:root:Oops Connection error', log.output)

if __name__ == '__main__':
    unittest.main()
