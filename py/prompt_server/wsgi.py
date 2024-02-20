#! /usr/bin/env python3

"""Start WSGI application"""

from app import app as application

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=8001)
