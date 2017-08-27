import os
import json
import dash
import copy
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

from interface import data
from server import app
from background import group_by_pointing_new

import numpy as np
import pandas as pd
import nicerlab as ni

from astropy.io.fits import getheader
from astropy.table import Table
from nicerlab import Eventlist
from support import generate_event_frame

def rebin_lc(arr, factor):
    return ni.lightcurve.Lightcurve(
            np.array([np.sum(i)/len(i) for i in np.split(arr, range(factor, arr.size, factor))]),
            dt=factor, tstart=arr.tstart, tstop=arr.tstop, mjd=arr.mjd)

cmap = ['rgb(141,211,199)', 'rgb(179,179,179)', 'rgb(190,186,218)', 
        'rgb(251,128,114)', 'rgb(128,177,211)', 'rgb(253,180,98)', 
        'rgb(179,222,105)']



# Get the pointing indices
mk_pointings = group_by_pointing_new(data.mkf['TIME'], factor=1.1)

# Build the pointing-time interval list
pti = [[data.mkf['TIME'][i0],data.mkf['TIME'][i1-1]] for i0,i1 in mk_pointings]
allpti = Table(rows=pti, names=['START', 'STOP'])

gtitable = Table(rows=data.gti, names=['START', 'STOP'])
gtitable['DURATION'] = gtitable['STOP'] - gtitable['START']

def float_or(string, default):
    if string != "":
        return float(string)
    else:
        return default

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe[col][i]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

def generate_gtilist(gti, max_rows=10):
    options = [{
                'label': "[{} ;{}]".format(t0,t1),
                'value': i
            } for i,[t0,t1] in enumerate(gti) ]

    # Truncate
    if len(options) > max_rows:
        options = options[:max_rows]

    return dcc.Checklist(
            id='gti-selection',
            options=options,
            values=[i for i in range(len(options))]
            )

# Collect the eventlist for all MPUs
tstart  = data.evt['TIME'].min()
tstop   = data.evt['TIME'].max()
origin = [data.mkf['TIME'][i0] for i0,i1 in mk_pointings]

gti = np.array(pti)

# Construct the light curve per MPU
etable_per_mpu = data.evt.group_by('MPU')
lc_per_mpu = [[ni.make_light_curve(np.sort(mpu['TIME']),dt=1,tstart=t0,tstop=t1)
                    for t0,t1 in pti] 
                for mpu in etable_per_mpu.groups]



#
# Configure the SCIENCE page
#
layout = html.Div(style={'margin': '20'}, children=[
    html.Hr(),

    html.H3('Science', style={'text-align': 'center'}, className='row'),

    # 3rd row with MPU/Detector filtering controls
    dcc.Dropdown(id='mpu-selection2',
        options=[
                {'label': "MPU{}".format(i), 'value': i} for i in range(7)
            ],
            multi=True,
            value=[i for i in range(7)]
    ),

    html.Label('MPU mode:'),
    dcc.RadioItems(id='mpu-mode',
        options=[
            {'label': 'sum', 'value': 'sum'},
            {'label': 'compare', 'value': 'compare'}
            ],
        value='compare',
        labelStyle={'display': 'inline-block'}
    ),
    html.Label('grid layout:'),
    dcc.RadioItems(id='panel-lc-mode',
        options=[
            {'label': 'regular', 'value': 'regular'},
            {'label': 'scaled' , 'value': 'scaled'}
            ],
        value='scaled',
        labelStyle={'display': 'inline-block'}
    ),
    

    #html.Div( className='row', children=[

        #html.Div( className='six columns', children=[
            #html.P('Filter on MPU:'),
            #dcc.Dropdown(id='mpu-selection',
                #options=[
                    #{'label': "MPU{}".format(i), 
                     #'value': 'mpu{}'.format(i)} for i in range(7)],
                #multi=True,
                #value=['mpu{}'.format(i) for i in range(7)]
            #),
            #dcc.RadioItems(id='mpu-mode',
                #options=[
                    #{'label': 'sum', 'value': 'sum'},
                    #{'label': 'compare', 'value': 'compare'}
                    #],
                #value='compare',
                #labelStyle={'display': 'inline-block'}
            #)]
        #),
        
        #html.Div( className='six columns', children=[
            #html.P('Filter on detector:'),
            #dcc.Dropdown(id='detector-selection',
                #options=[{'label': "D{}".format(i), 'value': i} for i in range(8)],
                #multi=True,
                #value=[i for i in range(8)]
            #)]
        #),
        
    #]),

    # 4th row with GTI controls
    #html.Br(),
    #html.Div(className='row', children=[

        #html.Div(className='five columns', children=[
            #html.H5('Good Time Interval'),

            #html.Div(className='row', children=[
                #html.Div(className='three columns', children=html.P('time offset: ')),
                #html.Div(className='nine columns', children=
                    #dcc.RadioItems(
                        #id='ti-offset',
                        #options=[
                            #{'label': 'absolute', 'value': 0},
                            #{'label': 'relative', 'value': origin}
                            #],
                        #value=origin,
                        #labelStyle={'display': 'inline-block'}
                    #),
                #),
            #]),

            #html.Div(className='row', children=[
                #html.Div(className='three columns', children=html.P('Interval type:')),
                #html.Div(className='nine columns', children=
                    #dcc.RadioItems(
                        #id='ti-type',
                        #options=[
                            #{'label': 'good', 'value': 'good'},
                            #{'label': 'bad', 'value': 'bad'}
                            #],
                        #value='good',
                        #labelStyle={'display': 'inline-block'}
                    #)
                #)
            #]),

            #html.Div(className='row', children=[
                #html.Div(className='three columns', children=html.P('merge logic: ')),
                #html.Div(className='nine columns', children=
                    #dcc.RadioItems(
                        #id='ti-logic',
                        #options=[
                            #{'label': 'and', 'value': 'and'},
                            #{'label': 'or' , 'value': 'or' }
                            #],
                        #value='and',
                        #labelStyle={'display': 'inline-block'}
                    #)
                #)
            #]),

            #html.Div(className='row', children=[
                #dcc.Input(id='ti-start',
                    #placeholder="tstart...",
                    #type='number',
                    #value='',
                    #style={'width': 180}
                #),
                #dcc.Input(id='ti-stop',
                    #placeholder="tstop...",
                    #type='number',
                    #value='', 
                    #style={'width': 180}
                #)
            #]),
            #html.Div(className='row', children=[
                #html.Button('merge interval', id='add-good-time', n_clicks=0,
                    #className="button-primary", style={'width': 180})
            #])
            #]),

        #html.Div([
            #html.Div(children=[
                #html.Div(id='gti-table', children=generate_gtilist(gti))
                #])
            #], className='four columns'
        #)
    #]),

    # 5th row with light curve
    html.Br(),
    dcc.Graph(id='panel-lc'),
    dcc.Slider(
        id='panel-lc-slider',
        min=1,max=16,
        value=4,
        step=None,
        marks={str(i): str(i) for i in [1,2,4,8,12,16]}
    ),

    
    # 5th row with light curve and linked PI histogram
    html.Br(),
    html.Br(),
    html.Div(className='row', children=[
        # Add light curve
        #html.Div(className='seven columns', children=[
            #dcc.Graph(id='graph-with-slider', animate=False),
            #dcc.Slider(
                #id='year-slider',
                #min=0.1,
                #max=16,
                #value=4,
                #step=None,
                #marks={str(year): str(year) for year in [0.1,0.5,1,2,4,8,16]}
                #)
            #]
        #),
        html.Div(className='six columns', children=[
            html.Br(),
            html.H5("TI selection:"),
            html.Div(id='ti-selection'),
            html.H5("GTI table:"),
            generate_table(gtitable)
        ]),

        # Add linked PI histogram
        html.Div(className='five columns', children=[
            dcc.Graph(id='linked-pi'),
            dcc.Input(id='lower-pi', type='number', value=60, style={'min': '0', 'max': '1500', 'width': '100'}),
            dcc.Input(id='upper-pi', type='number', value=800, style={'min': '0', 'max': '1500', 'width': '100'}),
            dcc.Input(id='delta-pi', type='number', value=3, style={'min': '0', 'max': '1500', 'width': '100'})
            ]
        )
    ]),

    # Add horizontal rule
    html.Hr(),

    # Add empty div
    html.Div(id='buffer', style={'height': '140'}),
    html.Div(id='gti-data', style={'display': 'none'},
                 children=json.dumps(gti.tolist())),

    html.Div(id='science-foot', style={'height':'80px'})
])



@app.callback(Output('panel-lc', 'figure'),
        [Input('mpu-selection2','value'),
         Input('mpu-mode', 'value'),
         Input('panel-lc-slider', 'value'),
         Input('panel-lc-mode', 'value')])
def make_panel_lc(mpu_selection, mpu_mode, bin_factor, panel_mode):
    # Allocate an empty array of traces
    traces = []

    # Allocate the axis keys
    xnames = ['x{}'.format(i+1) for i in range(len(gti)*1)]

    # Find the origins
    # origin = [data.mktable['TIME'][i0] for i0,i1 in mk_pointings]

    # Rebin the light curves
    blc_per_mpu = [[rebin_lc(lc,bin_factor) for lc in mpu] 
                                for mpu in lc_per_mpu]

    # Sum all selected MPU's
    lc_per_gti = np.sum([blc_per_mpu[i] for i in mpu_selection], axis=0)

    # Construct the light curve traces
    for n,[t0,t1] in enumerate(gti):
        # Construct single-mpu traces
        if mpu_mode == 'compare':
            # Create a trace and add to plot
            for i in mpu_selection:
                mpu_lc = blc_per_mpu[i]
                gti_trace = go.Scatter(
                        x=mpu_lc[n].timespace()-origin[n],
                        y=mpu_lc[n],
                        mode='lines',
                        name='MPU {}'.format(i),
                        line=dict(width=1.0, color=(cmap[i])),
                        legendgroup="{}".format(i),
                        xaxis=xnames[n],
                        yaxis='y'
                        )
                if n > 0: gti_trace['showlegend'] = False
                traces.append(gti_trace)

        # Construct summed-mpu trace
        if len(mpu_selection) > 0:
            gti_trace = go.Scatter(
                    x=blc_per_mpu[0][n].timespace()-origin[n],
                    y=lc_per_gti[n],
                    mode='lines+markers',
                    name='all data',
                    marker=dict(color='rgb(31,119,180)', size=0.1),
                    legendgroup='gti',
                    xaxis=xnames[n],
                    yaxis='y'
                )
            if n > 0: gti_trace['showlegend'] = False
            traces.append(gti_trace)
        
    # Construct the layout
    mylayout = go.Layout(
            legend=dict(orientation="h", x=-.1, y=1.2),
            yaxis=dict(title='ct/s'),
            dragmode='select',
        )

    # Compute the grid ratio
    gti_fraction = ni.gtitools.durations(gti) / np.sum(ni.gtitools.durations(gti))
    if panel_mode == 'regular':
        gti_fraction = np.full(len(gti), 1.0/len(gti))
    grid_bounds = np.cumsum(np.concatenate(([0],gti_fraction)))
    x_start = np.copy(grid_bounds[:-1])
    x_stop  = np.copy(grid_bounds[1:])
    # Apply spacing
    spacing = 0.025
    x_start[1:] += spacing
    x_stop[:-1] -= spacing

    # Set horizontal layout
    for n,xax in enumerate(xnames):
        mylayout['xaxis{}'.format(n+1)]=dict(
                domain=[x_start[n], x_stop[n]],
                title='time <br> (+{:.0f} s)'.format(origin[n])
            )

    # Set the vertical layout
    #for n,xax in enumerate(xnames):
        #layout['yaxis{}'.format(n+1)]=dict(domain=[0, 1], anchor=xax)

    return {
        'data': traces,
        'layout': mylayout
    }



#@app.callback(
        #dash.dependencies.Output('buffer', 'children'),
        #[dash.dependencies.Input('mpu-selection', 'value'),
         #dash.dependencies.Input('mpu-mode', 'value')])
#def update_mpu_selection(mpu_selection, mpu_mode):
    #print("> Updating mpu selection...")
    ## Fetch global memory
    #global evt_all, evt_mpu, df_mpu


    #print("> mpu_selection() completed")
    #return ''




# Add light curve interactivity
#@app.callback(
    #dash.dependencies.Output('graph-with-slider', 'figure'),
    #[dash.dependencies.Input('year-slider', 'value'),
     #dash.dependencies.Input('gti-selection', 'values'),
     #dash.dependencies.Input('mpu-selection', 'value'),
     #dash.dependencies.Input('detector-selection', 'value'),
     #dash.dependencies.Input('mpu-mode', 'value')],
    #state=[State('gti-data', 'children')]
    #)
#def update_figure(year, gti_selection, mpu_selection, det_selection, mpu_mode, gti_data):
    ## Grab the current gti
    #if gti_data != "":
        #gti = np.array(json.loads(gti_data))

    #print("> Updating figure data | dt = ", year)
    ##print("                       |gti = ", gti_selection)
    ##print("                       |mpu = ", mpu_selection)
    ##print("                       |mpu = ", repr(mpu_mode))
    ##print("                       |det = ", det_selection)
    ##print("                       |gti = ")
    ##print(gti[gti_selection])
    ##print("---")
    ##print(gti)

    ## Ensure data is selected
    #if len(gti_selection) is 0 or len(mpu_selection) is 0:
        #return {
                #'data': [],
                #'layout': go.Layout(
                    ## title="light curve",
                    #showlegend=False,
                    #xaxis=dict(title= 'Time (s)'),
                    #yaxis=dict(title='Counts'),
                    #margin={'l': 60, 'b': 40, 't': 50, 'r': 20},
                    #hovermode='compare'
                    #)
                #}
    
    ## Apply mpu selection
    #df_mpu = df.loc[mpu_selection]

    ## Select all events
    #evt_all = np.sort(df_mpu['TIME'])
    ## Select events per MPU
    #evt_mpu = [np.sort(mpu['TIME']) for name,mpu in df_mpu.groupby(level=0)]

    ## Apply the DETECTOR cut
    ## mask = data['DET'] 
    ## data = data.loc[mask]
    #print("> DET selection not yet implemented")

    ## Construct the lc for ALL MPUs
    #lc_all = [ni.make_light_curve(evt_all, dt=year, tstart=t0, tstop=t1) 
              #for t0,t1 in gti[gti_selection]]

    ## Construct the trace
    #trace = [
            #go.Scatter(
                #name="All data",
                #legendgroup='{}'.format(7),
                #x=ts.timespace() - origin,
                #y=ts,
                #showlegend=(not bool(i)),
                #line=dict(color = ('rgb(22, 96, 167)'))
                #) for i,ts in enumerate(lc_all)
            #]

    ## Make the per-mpu light curves
    #trace_mpu = []
    #if mpu_mode != 'sum':
        #lc_mpu = [[ni.make_light_curve(evt, dt=year, tstart=t0, tstop=t1) 
                   #for t0,t1 in gti[gti_selection]]
                  #for evt in evt_mpu]

        ## Stack the mpu light curves
        #if mpu_mode == 'stack':
            #for i in range(1,len(lc_mpu)):
                #for j,seg in enumerate(lc_mpu[i]):
                    ## for k in range(i):
                    #lc_mpu[i][j] += lc_mpu[i-1][j]

        #trace_mpu = [
                #go.Scatter(
                    #name='{}'.format(mpu_selection[i]),
                    #legendgroup='mpu{}'.format(mpu_selection[i].upper()),
                    #showlegend=(not bool(j)),
                    #x=ts.timespace() - origin,
                    #y=ts,
                    #hoverinfo='none',
                    #line=dict(width=1.0, color = (cmap[i]))
                    #) for i,lc in enumerate(lc_mpu) for j,ts in enumerate(lc) 
                #]

    ## Get the largest value
    #ymax = np.max([np.max(ts) for ts in lc_all]) * 1.10

    ## Build the gti curve
    #gtix = np.array([[t0,t0,t1,t1] for t0,t1 in gti[gti_selection]]).flatten()
    #gtiy = np.array(len(gti)*[0,ymax,ymax,0])
    #gtitrace = go.Scatter(
            #x=gtix-origin,
            #y=gtiy,
            #fill='tozeroy',
            #hoverinfo='none',
            #showlegend=False,
            #line=dict(width=0.0,color='rgb(225,245,196)')
    #)

    #print("> light_curve_update() completed")

    #return {
        #'data': [gtitrace] + trace_mpu + trace,
        #'layout': go.Layout(
            ## title="light curve",
            ##legend=dict(orientation="h",x=0,y=1.2),
            #xaxis=dict(title= 'Time (s)'),
            #yaxis=dict(title='Counts'),
            #margin={'l': 60, 'b': 40, 't': 50, 'r': 20},
            #hovermode='compare'
        #)
    #}

@app.callback(Output('ti-selection','children'), [Input('panel-lc', 'selectedData')])
def update_ti_selection(placeholder):
    domain = [None, None, None]

    # Collect the xaxis names
    xnames = ['x'] + ["x{}".format(i) for i in range(2,len(gti)+1)]
    # Retrieve the domain
    if placeholder is not None and 'range' in placeholder:
        for n,xax in enumerate(xnames):
            if xax in placeholder['range']:
                domain = [x+origin[n] for x in placeholder['range'][xax]]

                break
        domain.append(domain[1]-domain[0])

    tb = Table(rows=[domain], names=['START', 'STOP', 'DURATION'],
            dtype=('i8', 'i8', 'i8'))
    
    return generate_table(tb)


@app.callback(
    Output('linked-pi', 'figure'),
    [dash.dependencies.Input('lower-pi', 'value'),
     dash.dependencies.Input('upper-pi', 'value'),
     dash.dependencies.Input('delta-pi', 'value')]
    )
def update_linked_pi(lower_pi, upper_pi, delta_pi):
    # Select data
    idx = np.where(np.logical_and(data.evt['PI'] > int(lower_pi), 
                                  data.evt['PI'] < int(upper_pi)))
    pi  = data.evt[idx]['PI']

    # Build figure
    return {
        'data': [go.Histogram(
            x=pi,
            xbins=dict(
                start=0,
                end=1500,
                size=delta_pi
            ))],
        'layout': go.Layout(
            showlegend=False,
            xaxis={'title': 'PI Channel'},
            yaxis={'title': 'Counts'},
            hovermode='compare'
        )
    }



#@app.callback(
        #Output('gti-data', 'children'),
        #inputs=[Input('add-good-time', 'n_clicks')],
        #state=[State('ti-start' , 'value'),
               #State('ti-stop'  , 'value'),
               #State('ti-offset', 'value'),
               #State('ti-logic' , 'value'),
               #State('ti-type'  , 'value'),
               #State('gti-data' , 'children')])
#def update_gti(n_clicks, ti_start, ti_stop, ti_offset, ti_logic, ti_type, gti_data):
    ## Grab the existing data
    #if gti_data != "":
        #gti = np.array(json.loads(gti_data))
        #print("in update: ", gti)

    ## Ensure we have a real click
    #if n_clicks is 0:
        #return json.dumps(gti.tolist())

    ## Debug
    #print("> You clicked a button!")
    #print("  ~ Count...: ", n_clicks)
    #print("  ~ Offset..: ", ti_offset)

    #start = float_or(ti_start, 0)
    #stop  = float_or(ti_stop, np.max(gti))

    #new_gti = np.array([[start, stop]]) + ti_offset
    #if ti_type == 'bad':
        #print("\n>> MODIFYING GTI")
        #print(">> old:", new_gti)
        #new_gti = ni.gtitools.bad_to_good(new_gti)
        #print(">> new:", new_gti)
    #print("  ~ new gti.: ", new_gti)
    #print("  ~ old gti.: ", gti)

    #gti = ni.gtitools.merge([gti, new_gti], method=ti_logic)
    #print('setting gti to', json.dumps(gti.tolist()))

    #return json.dumps(gti.tolist())

#@app.callback(Output('gti-table', 'children'), [Input('gti-data','children')])
#def update_gti_table(gti_data):
    #print("json gti:", gti_data)
    #if gti_data != "":
        #gti = np.array(json.loads(gti_data))
    #return generate_gtilist(gti)

