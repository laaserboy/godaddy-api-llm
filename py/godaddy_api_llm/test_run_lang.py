'''Run tests for GD functions'''

import json
import unittest
from unittest.mock import patch
import gd_fun
from io import StringIO

class TestGetDomainAvailability(unittest.TestCase):
    @patch('gd_fun.get_page')
    def test_get_domain_availability(self, mock_get_page):
        test_domain = "example1.com"
        mock_response = {'available': True}
        mock_get_page.return_value = mock_response
        gd_key = "your_gd_key"
        gd_secret = "your_gd_secret"
        result = gd_fun.get_domain_availability(test_domain, gd_key, gd_secret)
        self.assertEqual(result, mock_response)
        mock_get_page.assert_called_once_with(
            "/v1/domains/available",
            {},
            {"domain": test_domain}, 'your_gd_key', 'your_gd_secret'
        )

class TestGetSuggestions(unittest.TestCase):
    @patch('gd_fun.get_page')
    def test_get_suggestions(self, mock_get_page):
        mock_response = mock_get_page.return_value
        mock_response.json.return_value = [{"domain": "example.com"}]
        result = gd_fun.get_suggestions('example', 'Cupertino', 'key', 'secret')
        expected_result = json.dumps({"domain": "example.com"})
        self.assertEqual(result, expected_result)
        mock_response.json.return_value = []
        result = gd_fun.get_suggestions('What is a good domain?', 'Cupertino', 'key', 'secret')
        expected_empty_result = '{}'
        self.assertEqual(result, expected_empty_result)

if __name__ == '__main__':
    unittest.main()
