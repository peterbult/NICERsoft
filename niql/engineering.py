import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

from interface import data
from server import app

# Sum each detector stream
total_counts = np.sum(np.asarray(data.mkf['MPU_XRAY_COUNT']), axis=0)

# Allocate the focalplane image
focalplane = np.zeros((7,8))
# Populate the focalplane
for i,counts in enumerate(np.ravel(total_counts)):
    focalplane[data.det_coords[i]] = counts

# Allocate the focalplane hover label
focalhover = np.full((7,8), "disabled", dtype='<U32')
# Populate hover labels
for i,counts in enumerate(np.ravel(total_counts)):
    if counts > 0:
        focalhover[data.det_coords[i]] = '{} counts <br> DET {}'.format(
            counts,
            data.det_names[i]
        )

# Build the detector histogram
detector_histogram = go.Bar(
    x=data.det_names,
    y=np.ravel(total_counts)
)

# Build the deadtime histogram
deadtime_histogram = go.Histogram(
    x=data.evt['DEADTIME']
)

# Build the resets histogram
under_counts = np.sum(np.asarray(data.mkf['MPU_UNDER_COUNT']), axis=0)
resets_histogram = go.Bar(
    x=data.det_names,
    y=np.ravel(under_counts)
)

#
# Setup the ENGINEERING page
#
layout = html.Div(style={'margin': '20'}, children=[
    html.Hr(),

    html.H3('Engineering', style={'text-align': 'center'}, className='row'),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
                dcc.Graph(id='focalplane'),
                dcc.RangeSlider(id='focalplane-slider',
                    count=1,
                    min=0,
                    max=np.max(focalplane),
                    step=1,
                    value=np.array([0.8,1.0])*np.max(focalplane),
                    marks={i: '{}'.format(i) for i in 
                        range(0,int(np.max(focalplane)),int(np.max(focalplane)/10))},
                    className='hidden-print'
                )
            ]
        ),
        html.Div(className='six columns', children=
            dcc.Graph(id='detector-hist')
        )
    ]),

    html.Br(),
    html.Br(),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=
            dcc.Graph(id='resets-hist')
        ),
        html.Div(className='six columns', children=
            dcc.Graph(id='deadtime-hist')
        )
    ]),

    html.Div(id='engineering-foot', className='row', style={'height':'80px'})
])


@app.callback(Output('focalplane', 'figure'), [Input('focalplane-slider', 'value')])
def update_focalplane(slider_value):
    return {
        'data': [go.Heatmap(
            z=np.clip(focalplane, *slider_value), 
            colorscale='Portland',
            text=focalhover,
            hoverinfo='text'
        )], 
        'layout': go.Layout(
            title='focal plane', 
            xaxis=dict(title='rawx'),
            yaxis=dict(title='rawy', autorange='reversed')
            )
        }

@app.callback(Output('detector-hist','figure'),[Input('focalplane-slider', 'value')])
def update_detector_hist(slider_value):
    return {
    'data': [detector_histogram],
    'layout': go.Layout(
        title='detector histogram',
        xaxis=dict(title='Detector ID'),
        yaxis=dict(title='Counts', range=slider_value)
        )
}

@app.callback(Output('resets-hist','figure'),[Input('engineering-foot', 'children')])
def update_resets_hist(slider_value):
    return {
    'data': [resets_histogram],
    'layout': go.Layout(
        title='reset rate histogram',
        xaxis=dict(title='Detector ID'),
        yaxis=dict(title='Counts')
        )
}

@app.callback(Output('deadtime-hist','figure'),[Input('engineering-foot', 'children')])
def update_deadtime_hist(slider_value):
    return {
    'data': [deadtime_histogram],
    'layout': go.Layout(
        title='deadtime histogram',
        xaxis=dict(title='Timescale'),
        yaxis=dict(title='Counts', type='log')
        )
}

