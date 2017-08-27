# -*- coding: utf-8 -*-
import os
import dash
from flask_caching import Cache

app = dash.Dash(__name__, csrf_protect=False)
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': '/tmp/niql.cache'
})
app.config.supress_callback_exceptions = True

