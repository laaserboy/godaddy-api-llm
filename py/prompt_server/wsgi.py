#! /usr/bin/env python3

"""Start WSGI application"""

from app import app as application

if __name__ == "__main__":
    application.run(port=5001)
