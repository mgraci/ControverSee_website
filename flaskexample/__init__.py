# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask
app = Flask(__name__)
from flaskexample import views


if __name__ == '__main__':
    app.debug = True
    app.run()
