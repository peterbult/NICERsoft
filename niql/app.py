import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import flask

from interface import data
from server import app

import engineering
import background
import science
import trumpet
import home

# Extract the header
header = data.evt.meta
dir_path = os.path.dirname(os.path.realpath(__file__))

app.layout = html.Div([
    # URL bar, doesn't render anything
    dcc.Location(id='url', refresh=False),

    # top row with banner and logo
    html.Div([
        html.H1(children="NICER Quicklook", className='eight columns'),

        html.Img(
            src="https://upload.wikimedia.org/wikipedia/commons/6/63/NICER_-_SEXTANT_logo.png",
            className='four columns',
            style={'float': 'right', 'width': '150', 'position': 'relative'}),
        ],
        className='row'
    ),

    # 2nd row with statistics
    html.Div([
        html.H5(className='three columns', children="ObsID: {}".format(header['OBS_ID'])),
        html.H5(className='three columns', 
            children="Source: {}".format(header['OBJECT']),
            style={'text-align': 'center'}
        ),
        html.H5(className='three columns', 
            children="Exposure: {:.1f} ks".format(float(header['EXPOSURE'])/1000),            
            style={'text-align': 'center'}
        ),
        html.H5(className='three columns', 
            children="Counts: {}".format(len(data.evt)),
            style={'text-align': 'right'}
        )
        ],
        className='row'
    ),
    html.Hr(),

    # 3rd row with navigation buttons
    html.Div(className='row', children=[
        dcc.Link('Home', href='/', className='button'),
        dcc.Link('Engineering', href='/engineering', className='button'),
        dcc.Link('Background', href='/background', className='button'),
        dcc.Link('PI ratio', href='/pi_ratio', className='button'),
        dcc.Link('Science', href='/science', className='button')
    ]),

    # content will be rendered in this element
    html.Div(id='page-content')
])

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/engineering':
        return engineering.layout
    if pathname == '/background':
        return background.layout
    if pathname == '/pi_ratio':
        return trumpet.layout
    if pathname == '/science':
        return science.layout
    if pathname == '/':
        return home.layout

    return html.Div()


css_directory = os.path.join(os.getcwd(), 'data')
stylesheets = ['style.css']
static_css_route = '/static/'

@app.server.route('{}<stylesheet>'.format(static_css_route))
def serve_stylesheet(stylesheet):
    if stylesheet not in stylesheets:
        raise Exception(
            '"{}" is excluded from the allowed static files'.format(
                stylesheet
                )
            )
    return flask.send_from_directory(css_directory, stylesheet)

for stylesheet in stylesheets:
    app.css.append_css({"external_url": "/static/{}".format(stylesheet)})

if __name__ == '__main__':
    try: 
        app.run_server()
    except Exception as e:
        log.error(e)


