import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, Event
import plotly.graph_objs as go
import datetime
import numpy as np

from interface import data
from server import app

from scipy.sparse import coo_matrix

# Define a quick matrix histogram function
def crazy_histogram2d(x, y, nx, ny):
    data = np.ones(len(x))
    y = np.clip(y, 0, ny-1)
    x = np.clip(x, 0, nx-1)
    return coo_matrix((data, (y, x)), shape=(ny, nx)).toarray()

# Define some rebin utils
def rebin_x(arr, f):
    return np.sum(arr.reshape(arr.shape[0],-1,f), axis=2)
def rebin_y(arr, f):
    return np.sum(arr.reshape(-1,f,arr.shape[1]), axis=1)



# Define the ratio grid
rmin = 0
rmax = 3
rbins= 600
rres = rbins/(rmax-rmin)

# Grab the PI columns
mask = np.where(data.evt['PI_FAST']>0)

pi_table = data.evt['PI', 'PI_FAST', 'DET_ID'][mask].group_by('DET_ID')

# Format the PI column
pi_column = [
    np.array(det['PI'], dtype=np.intp) 
    for det in pi_table.groups
]

# Compute a fast to slow ratio column
fast_to_slow_column = [
    np.array(det['PI_FAST']/det['PI']*rres, dtype=np.intp) 
    for det in pi_table.groups
] 

# Compute the fast to slow heatmap
fast_to_slow_trumpet = np.array([
    crazy_histogram2d(pi, ratio, nx=1500, ny=rbins)
    for pi, ratio in zip(pi_column, fast_to_slow_column)
])

# Compute a slow to fast ratio column
slow_to_fast_column = [
    np.array(det['PI']/det['PI_FAST']*rres, dtype=np.intp) 
    for det in pi_table.groups
] 

# Compute the slow to fast heatmap
slow_to_fast_trumpet = np.array([
    crazy_histogram2d(pi, ratio, nx=1500, ny=rbins)
    for pi, ratio in zip(pi_column, slow_to_fast_column)
])


# Extract the relevant data in a more convenient format
# mask = np.where(data.etable['PHA_FAST']>0) #np.invert(data.etable['PHA_FAST'][-32768])
# tb_grp = data.etable['PI', 'PHA', 'PHA_FAST', 'MPU'][mask].group_by('MPU')
# #tb_grp = data.etable['PI', 'PHA', 'PHA_FAST', 'MPU'].group_by('MPU')
# pi_column = [np.array(tb['PI'], dtype=np.intp) for tb in tb_grp.groups]
# sf_column = [np.array(np.trunc(np.array(tb['PHA'],dtype=float) 
#                             / np.array(tb['PHA_FAST'], dtype=float)
#                             * rres
#                             ), dtype=np.intp)
#                         for tb in tb_grp.groups]
# fs_column = [np.array(np.trunc(np.array(tb['PHA_FAST'],dtype=float) 
#                             / np.array(tb['PHA'], dtype=float)
#                             * rres
#                             ), dtype=np.intp) 
#                         for tb in tb_grp.groups]
# 
# # Compute the base SLOW / FAST
# sf_trumpet = np.array([crazy_histogram2d(pi, sf,
#                                nx=1500, ny=rbins)
#                 for pi, sf in zip(pi_column, sf_column)])
# 
# # Compute the base SLOW / FAST
# fs_trumpet = np.array([crazy_histogram2d(pi, fs,
#                                nx=1500, ny=rbins)
#                 for pi, fs in zip(pi_column, fs_column)])

# Prepare the selection menu
selection_labels = [
    'DET {}'.format(i) 
    for i in np.array(pi_table.groups.keys).astype(int)
]
selection_values = np.arange(len(selection_labels))


#
# Setup the TRUMPET page
#
layout = html.Div(style={'margin': '20'}, children=[
    html.Hr(),

    html.H3('PHA ratio', style={'text-align': 'center'}, className='row'),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            html.Label('ratio mode: '),
            dcc.RadioItems(
                id='ratio-mode',
                options=[
                        {'label': 'fast / slow', 'value': 'fast/slow'},
                        {'label': 'slow / fast', 'value': 'slow/fast'}
                    ],
                value='slow/fast',
                labelStyle={'display': 'inline-block'}
            ),

            html.Label('count mode: '),
            dcc.RadioItems(
                id='count-mode',
                options=[
                        {'label': 'linear', 'value': 'lin'},
                        {'label': 'logarithmic', 'value': 'log'}
                    ],
                value='lin',
                labelStyle={'display': 'inline-block'}
            )
        ]),

        html.Div(className='six columns', children=[
            html.Label('PI bins: '),
            dcc.Slider(id='sf-pi-bins',     
                min=1,max=20,step=None,
                marks={i: '{}'.format(i) for i in [1,2,3,4,5,6,10,12,15,20]},
                value=10
            ),
            html.Br(),
            html.Label('RATIO bins: '),
            dcc.Slider(id='sf-ratio-bins',     
                min=1,max=20,step=None,
                marks={i: '{}'.format(i) for i in [1,2,3,4,5,6,10,12,15,20]},
                value=3
            ),
            html.Br()
        ]),
    ]),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
            html.P(''),
            #html.P('Filter on DET:'),
            #dcc.Dropdown(id='trumpet-det',
                #options=[
                    #{'label': "DET{}".format(i), 
                     #'value': i} for i in range(8)],
                #multi=False,
                #value=[i for i in range(7)]
            #)
        ]),
        html.Div(className='six columns', children=[
            html.P('Filter on MPU:'),
            dcc.Dropdown(id='trumpet-mpu',
                options=[
                    {'label': label, 
                     'value': value} 
                     for label,value in zip(selection_labels, selection_values)
                ],
                multi=True,
                value=selection_values
            )
        ])
    ]),

    html.Div(className='row', children=[
        html.Div(className='six columns', children=[
                dcc.Graph(id='trumpet-1')
            ]
        ),
        html.Div(className='six columns', children=
            dcc.Graph(id='trumpet-2')
        )
    ]),


    html.Div(id='trumpet-foot', className='row', style={'height':'80px'})
])

@app.callback(Output('trumpet-1', 'figure'), 
        [Input('sf-pi-bins', 'value'),
         Input('sf-ratio-bins', 'value'),
         Input('count-mode', 'value'),
         Input('ratio-mode', 'value')])
def update_trumpet(pi_bin, ratio_bin, count_mode, ratio_mode):
    ratio_bin = int(ratio_bin)
    pi_bin = int(pi_bin)
    
    image = []
    if ratio_mode == 'slow/fast':
        image = np.sum(slow_to_fast_trumpet, axis=0)
    elif ratio_mode == 'fast/slow':
        image = np.sum(fast_to_slow_trumpet, axis=0)


    image = rebin_y(rebin_x(image, pi_bin), ratio_bin)

    hovertext = np.copy(image)

    if count_mode == 'log':
        image[image==0]+=1
        image = np.log10(image)

    return {
        'data': [go.Heatmap(
            z=image,
            y=np.linspace(0,3,image.shape[0]),
            x=np.linspace(0,1500,image.shape[1]),
            text=hovertext,
            hoverinfo='text',
            colorscale='Portland'
        )], 
        'layout': go.Layout(
            title='all data', 
            xaxis=dict(title='PI'),
            yaxis=dict(title=ratio_mode)
            )
        }

@app.callback(Output('trumpet-2', 'figure'), 
        [Input('sf-pi-bins', 'value'),
         Input('sf-ratio-bins', 'value'),
         Input('count-mode', 'value'),
         Input('ratio-mode', 'value'),
         Input('trumpet-mpu', 'value')])
def update_trumpet2(pi_bin, ratio_bin, count_mode, ratio_mode, mpu_selection):
    ratio_bin = int(ratio_bin)
    pi_bin = int(pi_bin)

    image = []
    if ratio_mode == 'slow/fast':
        #image = np.sum(sf_trumpet[mpu_selection], axis=0)
        image = np.sum(slow_to_fast_trumpet[mpu_selection], axis=0)
    elif ratio_mode == 'fast/slow':
        image = np.sum(fast_to_slow_trumpet[mpu_selection], axis=0)
        #image = np.sum(fs_trumpet[mpu_selection], axis=0)


    image = rebin_y(rebin_x(image, pi_bin), ratio_bin)

    hovertext = np.copy(image)

    if count_mode == 'log':
        image[image==0]+=1
        image = np.log10(image)

    return {
        'data': [go.Heatmap(
            z=image,
            y=np.linspace(0,3,image.shape[0]),
            x=np.linspace(0,1500,image.shape[1]),
            text=hovertext,
            hoverinfo='text',
            colorscale='Portland'
        )], 
        'layout': go.Layout(
            title='selection', 
            xaxis=dict(title='PI'),
            yaxis=dict(title=ratio_mode)
            )
        }

