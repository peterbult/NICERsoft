import dash
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import os

from interface import data
from server import app


intro = '''
Welcome to niql ( _/'nik(ə)l/_ ),

Build on masterplotter, niql is an interactive quicklook tool
for NICER observations. 

**Warning: niql is not a science tool**. While there is a science
tab, niql is not intented to produce science products. Instead it
let 

Hover, click and zoom to shape the data any way you like.

Some tips:
* click on the axis to manually set the range
* hover over a data point to see its properties
* double click to zoom out
* in the science tab you can select a portion of the light curve
  to see its time interval
* all time axes show time since the start of the respective continuous
  pointing.
* To save a plot, hover and click the camera icon
* To save the page, simply print to pdf
'''

#
# Setup the HOME page
#
layout = html.Div(style={'margin': '20'}, children=[
    html.Hr(),

    html.H3('Home', style={'text-align': 'center'}, className='row'),

    dcc.Markdown(intro),
    
    html.Div(id='home-foot', className='row', style={'height':'80px'})
])

