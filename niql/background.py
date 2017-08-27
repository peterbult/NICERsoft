import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objs as go
import os
from plotly import tools

from interface import data
from server import app

import numpy as np
import nicerlab as ni
from astropy.table import Table
#from nicerlab.utils import utils


# Define a color scheme for day/night
DayNight = [
    [0, 'rgb(12,51,131)'], [0.5, 'rgb(12,51,131)'],
    [0.5, 'rgb(242,211,56)'], [1, 'rgb(242,211,56)']
]

red    = 'rgb(214,39,40)'
blue   = 'rgb(31,119,180)'
orange = 'rgb(255,127,14)'
green  = 'rgb(44,160,44)'
purple = 'rgb(148,103,189)'
grey   = 'rgb(127,127,127)'
pink   = 'rgb(227,119,194)'
brown  = 'rgb(140,86,75)'


# Define a function that find the start and stop indices
# of continuous poitings in a light curve
def group_by_pointing(time, factor=1, allgti=None):
    steps = np.diff(time)
    width = steps[0]
    breaks = np.argwhere(steps>width*factor)

    start, stop = [], []
    if len(breaks) == 0:
        start = [0]
        stop = [len(time)]
    else:
        breaks = breaks.flatten() + 1

        start = np.concatenate(([0], breaks))
        stop  = np.concatenate((breaks,[len(time)]))

    pointing_indices = np.array([start,stop]).T

    if allgti is not None:
        if len(start) != len(allgti):
            # Extract the GTI times
            start_times = allgti['START']
            stop_times  = allgti['STOP']

            # Make a pointing time-interval
            pti =[[time[i0], time[i1-1]] for i0,i1 in pointing_indices]

            # Find the GTI indices associated with each PTI entry
            indices = ni.gtitools.apply_gti(start_times, pti)

            # Process is there is more than 1 GTIs in a PTI entry
            for row in indices:
                if len(row) > 1:
                    for n in row[:-1]:
                        # Calculate the middle time between the
                        # two adjacent GTIs
                        first = allgti['STOP'][n]
                        final = allgti['START'][n]
                        middle = (first+final)/2

                        # Find the TIME index closest to that middle-time
                        new_idx = ni.utils.find_first_of(time, middle)

                        # Add an extra break at the obtained index
                        start = np.insert(start, 0, new_idx)
                        stop  = np.insert(stop , 0, new_idx)

    # Add the additional GTI breaks
    start = np.sort(start)
    stop  = np.sort(stop)
    pointing_indices = np.array([start,stop]).T

    return pointing_indices


def group_by_pointing_new(time, factor=1, pti=None):
    steps = np.diff(time)
    width = steps[0]
    breaks = np.argwhere(steps>width*factor)

    start, stop = [], []
    if len(breaks) == 0:
        start = [0]
        stop = [len(time)]
    else:
        breaks = breaks.flatten() + 1

        start = np.concatenate(([0], breaks))
        stop  = np.concatenate((breaks,[len(time)]))

    ti_indices = np.array([start,stop]).T

    if pti is not None:
        if len(start) != len(pti):
            # Extract the pti times
            pti_start_times = pti['START']
            pti_stop_times  = pti['STOP']

            # Make a local time-interval table
            lti =[[time[i0], time[i1-1]] for i0,i1 in ti_indices]

            # Apply the table ti on the start times
            indices = ni.gtitools.apply_gti(pti_start_times, lti)

            # Process is there is more than 1 GTIs in a PTI entry
            for row in indices:
                if len(row) > 1:
                    for n in row[:-1]:
                        # Calculate the middle time between the
                        # two adjacent GTIs
                        first = pti['STOP'][n]
                        final = pti['START'][n]
                        middle = (first+final)/2

                        # Find the TIME index closest to that middle-time
                        new_idx = ni.utils.find_first_of(time, first)

                        # Add an extra break at the obtained index
                        start = np.insert(start, 0, new_idx)
                        stop  = np.insert(stop , 0, new_idx)

    # Add the additional GTI breaks
    start = np.sort(start)
    stop  = np.sort(stop)
    pointing_indices = np.array([start,stop]).T

    return pointing_indices


# Get the HKFILE pointing indices
# hk_pointings = group_by_pointing(data.hkmet,factor=1.1, allgti=data.allgti)
# hk_pointings = group_by_pointing_new(data.hkmet,factor=1.05)

# Build the pointing-time interval list
# pti = [[data.hkmet[i0],data.hkmet[i1-1]] for i0,i1 in hk_pointings]
# allpti = Table(rows=pti, names=['START', 'STOP'])

# Get the MKTABLE pointing indices
mk_pointings = group_by_pointing_new(data.mkf['TIME'], factor=1.1)
pti = [[data.mkf['TIME'][i0],data.mkf['TIME'][i1-1]] for i0,i1 in mk_pointings]

# if gtirows is not None:
    # mk_pointings = mk_pointings[gtirows]

# if gtirows is not None:
    # hk_pointings = hk_pointings[gtirows]

# Get the origins
# origin = [data.mktable['TIME'][i0] for i0,i1 in mk_pointings]
# origin = [data.hkmet[i0] for i0,i1 in hk_pointings]
origin = [data.mkf['TIME'][i0] for i0,i1 in mk_pointings]


# Read the SAA
dirname, filename = os.path.split(os.path.abspath(__file__))
saa_coords = Table.read(os.path.join(dirname, '../data/saa_lonlat.txt'), 
                        format='ascii', names=['LON','LAT'])
sph_coords = Table.read(os.path.join(dirname, '../data/sph.niql.txt'), 
                        format='ascii', names=['LON','LAT'])
nph_coords = Table.read(os.path.join(dirname, '../data/nph.niql.txt'), 
                        format='ascii', names=['LON','LAT'])



# Construct the flight path figure
saa_path = dict(
    type = 'scattergeo',
    lon = np.array(saa_coords['LON']),
    lat = np.array(saa_coords['LAT']),
    mode = 'lines',
    name = 'SAA',
    line = dict(width = 2, color = 'black')
)
sph_path = dict(
    type = 'scattergeo',
    lon = np.array(sph_coords['LON']),
    lat = np.array(sph_coords['LAT']),
    mode = 'lines',
    name='SPH',
    line = dict(width = 2, color = 'black')
)
nph_path = dict(
    type = 'scattergeo',
    lon = np.array(nph_coords['LON']),
    lat = np.array(nph_coords['LAT']),
    mode = 'lines',
    name='NPH',
    line = dict(width = 2, color = 'black')
)

# Extract the satellite coordinates
lon = data.mkf['SAT_LON']
lat = data.mkf['SAT_LAT']

# Construct the flight path figure
flight_path = [dict(
    type = 'scattergeo',
    lon = lon[i0:i1],
    lat = lat[i0:i1],
    name = 'ISS | SEG {}'.format(n),
    text = ["t: {:.2f} <br> ({:.1f}, {:.1f})".format(t-origin[n], lo, la) 
        for t,lo,la in zip(data.mkf['TIME'][i0:i1], lon[i0:i1], lat[i0:i1])],
    #hoverinfo='text+x+y',
    mode = 'markers',
    hoverinfo = 'markers+text+name',
    marker = dict(width = 2, color = np.arange(len(lon)), 
        colorscale='Portland', colorbar=go.ColorBar(title='Time'))
) for n,[i0,i1] in enumerate(mk_pointings)]


# Extract the ratio data
# ratio = np.array(data.etable['PHA'])/np.array(data.etable['PHA_FAST'])
# ratio_rejected = np.array(data.etable[np.where(ratio > 1.4)[0]]['TIME'])
ratio_events = data.evt['PI_RATIO']
ratio_rejected = np.array(data.evt[np.where(ratio_events > 1.4)[0]]['TIME'])


#
# Setup the BACKGROUND page
#
layout = html.Div(style={'margin': '20'}, children=[
    html.Hr(),

    html.H3('Background', style={'text-align': 'center'}, className='row'),

    html.Div(className='row', 
        style={'page-break-after': 'always', 'text-align': 'center'},
        children=[
        html.Div(className='three columns', children=[
            html.P('Map type:'),
            dcc.Dropdown(id='map-type',
                options=[
                    {'label': 'Equirectangular', 'value': 'equirectangular'},
                    {'label': 'Orthographic', 'value': 'orthographic'},
                    {'label': 'Robinson', 'value': 'robinson'}
                ],
                value='robinson'
            ),

            html.P('Overlay:'),
            dcc.Dropdown(id='overlay-type',
                options=[
                    {'label': 'Pointing', 'value': 'ANG_DIST'},
                    {'label': 'Sunshine', 'value': 'SUNSHINE'},
                    {'label': 'Time', 'value': 'TIME'}
                ],
                value='TIME'
            )
        ]),

        html.Div(className='nine columns', children=
            dcc.Graph(id='iss-flight')
        )
    ]),

    html.Br(),
    html.Label('grid mode:'),
    dcc.RadioItems(id='panel-mode',
        options=[
            {'label': 'regular', 'value': 'regular'},
            {'label': 'scaled',  'value': 'scaled'}
            ],
        value='scaled',
        labelStyle={'display': 'inline-block'}
    ),

    html.Div(className='row',
        style={'page-break-after': 'always', 'text-align': 'center'},
        children=[dcc.Graph(id='panel-graph')]
    ),

    html.Div(id='background-foot', className='row', style={'height':'80px'})
])


@app.callback(Output('iss-flight', 'figure'), 
             [Input('map-type', 'value'),
              Input('overlay-type', 'value')])
def update_iss_flight(projection, overlay):
    # Reset the colorscale
    for n,[fp,[i0,i1]] in enumerate(zip(flight_path,mk_pointings)):
        fp['marker']['colorscale'] = 'Portland'
        # Update flight path formatting
        if overlay == 'ANG_DIST':
            # Special case for pointing angle
            fp['marker'] = dict(width = 2, colorscale='Portland',
                color = np.log10(np.array(data.mkf[overlay][i0:i1])))
            if n == 0:
                fp['marker'].update(colorbar=go.ColorBar(title='Log[{}]'.format(overlay)))
        elif overlay == 'SUNSHINE':
            # Special color scale for SUNSHINE
            fp['marker'] = dict(width = 2, colorscale=DayNight,
                    color = np.array(data.mkf[overlay][i0:i1]))
            if n == 0:
                fp['marker'].update(colorbar=go.ColorBar(title=overlay),
                                    cmin=0, cmax=1)
        elif overlay == 'TIME':
            # Default case for time
            fp['marker'] = dict(width = 2, colorscale='Portland',
                    color = np.array(data.mkf[overlay][i0:i1]-origin[n]))
            if n == 0:
                fp['marker'].update(colorbar=go.ColorBar(title=overlay))

    flight_layout = dict(
            title = 'satellite flight path',
            showlegend = False, 
            geo = dict(
                scope='world',
                showland = True,
                landcolor = 'rgb(243, 243, 243)',
                countrycolor = 'rgb(204, 204, 204)',
                projection=dict(type=projection)
                )
            )

    return {
            'data': flight_path + [saa_path, sph_path, nph_path],
            'layout': flight_layout
            }


@app.callback(Output('panel-graph', 'figure'), [Input('panel-mode', 'value')])
def make_panel_plot(panel_mode):
    # Allocate an empty array of traces
    traces = []

    # Allocate the axis keys
    xnames = ['x{}'.format(i+1) for i in range(len(pti)*1)]
    ynames = ['y{}'.format(i+1) for i in range(len(pti)*6)]

    # Find the origins
    # origin = [data.mktable['TIME'][i0] for i0,i1 in mk_pointings]
    # origin = [data.hkmet[i0] for i0,i1 in hk_pointings]

#    # Construct the PHA-ratio trace
#    for n,[t0,t1] in enumerate(pti):
#        # Create the 1-sec light curve
#        ratio_curve = ni.make_light_curve(ratio_rejected, dt=1, tstart=t0, tstop=t1)
#        # Grab the time axis
#        time = ratio_curve.timespace() - origin[n]
#        # Build the light curve
#        ratio_trace = go.Scatter(
#                x=time,
#                y=ratio_curve,
#                mode='markers',
#                name='PHA ratio',
#                marker=dict(color=red), 
#                legendgroup='ratio',
#                xaxis=xnames[n],
#                yaxis=ynames[5+n*6]
#            )
#        if n > 0: ratio_trace['showlegend'] = False
#
#        traces.append(ratio_trace)
#
#    # Construct the traces from HKFILES
#    for n,[i0,i1] in enumerate(hk_pointings):
#        # Grab the time axis
#        time = data.hkmet[i0:i1] - origin[n]
#
#        # Find the first positive time
#        first_idx = ni.utils.find_first_of(time, 0)
#        
#
#        # Build the light curves
#        over_curve = np.array(data.hkovershoots[i0:i1])
#        over_trace = go.Scatter(
#                x=time[first_idx:],
#                y=over_curve[first_idx:],
#                mode='markers',
#                name='overshoot',
#                marker=dict(color=brown), 
#                legendgroup='overshoot',
#                xaxis=xnames[n],
#                yaxis=ynames[4+n*6]
#            )
#        if n > 0: over_trace['showlegend'] = False
#        traces.append(over_trace)
#
#        under_curve = np.array(data.hkundershoots[i0:i1])
#        under_trace = go.Scatter(
#                x=time[first_idx:],
#                y=under_curve[first_idx:],
#                mode='markers',
#                name='undershoot',
#                marker=dict(color=pink), 
#                legendgroup='undershoot',
#                xaxis=xnames[n],
#                yaxis=ynames[3+n*6]
#            )
#        if n > 0: under_trace['showlegend'] = False
#        traces.append(under_trace)


    # Construct the traces from MKF TABLE
    for n,[i0,i1] in enumerate(mk_pointings):    
        # Grab the time axis
        time = np.array(data.mkf['TIME'][i0:i1]) - origin[n]

        # Make the X-ray count light curve
        lc_curve = np.array(data.mkf['TOT_XRAY_COUNT'][i0:i1])
        lc_trace = go.Scatter(
            x=time,
            y=lc_curve,
            mode='markers',
            name='X-ray counts',
            marker=dict(color='black'),
            legendgroup='lc',
            xaxis=xnames[n],
            yaxis=ynames[5+n*6]
        ) 
        if n > 0: lc_trace['showlegend'] = False
        traces.append(lc_trace)

        # Make the under/over-shoot curves
        under_curve = np.array(data.mkf['TOT_UNDER_COUNT'][i0:i1])
        under_trace = go.Scatter(
            x=time,
            y=under_curve,
            mode='markers',
            name='undershoot',
            marker=dict(color=pink),
            legendgroup='undershoot',
            xaxis=xnames[n],
            yaxis=ynames[4+n*6]
        )
        if n > 0: under_trace['showlegend'] = False
        traces.append(under_trace)

        over_curve = np.array(data.mkf['TOT_OVER_COUNT'][i0:i1])
        over_trace = go.Scatter(
            x=time,
            y=over_curve,
            mode='markers',
            name='overshoot',
            marker=dict(color=brown),
            legendgroup='overshoot',
            xaxis=xnames[n],
            yaxis=ynames[4+n*6]
        )
        if n > 0: over_trace['showlegend'] = False
        traces.append( over_trace)

        # Make the pointing curves
        az_curve = np.array(data.mkf['ATT_ANG_AZ'][i0:i1])
        az_trace = go.Scatter(
            x=time,
            y=az_curve,
            mode='markers',
            name='azimuth',
            marker=dict(color=grey),
            legendgroup='azimuth',
            xaxis=xnames[n],
            yaxis=ynames[3+n*6]
        )
        if n > 0: az_trace['showlegend'] = False
        traces.append(az_trace)

        el_curve = np.array(data.mkf['ATT_ANG_EL'][i0:i1])
        el_trace = go.Scatter(
            x=time,
            y=el_curve,
            mode='markers',
            name='elevation',
            marker=dict(color=red),
            legendgroup='elevation',
            xaxis=xnames[n],
            yaxis=ynames[3+n*6]
        )
        if n > 0: el_trace['showlegend'] = False
        traces.append(el_trace)


        # Make the sun/moon/earth angle curves
        sun_curve = np.array(data.mkf['SUN_ANGLE'][i0:i1])
        sun_trace = go.Scatter(
                x=time,
                y=sun_curve,
                mode='markers',
                name='sun',
                marker=dict(color=orange), 
                legendgroup='sun',
                xaxis=xnames[n],
                yaxis=ynames[2+n*6]
            )
        if n > 0: sun_trace['showlegend'] = False
        traces.append(sun_trace)
        
        moon_curve = np.array(data.mkf['MOON_ANGLE'][i0:i1])
        moon_trace = go.Scatter(
                x=time,
                y=moon_curve,
                mode='markers',
                name='moon',
                marker=dict(color=blue), 
                legendgroup='moon',
                xaxis=xnames[n],
                yaxis=ynames[2+n*6]
            )
        if n > 0: moon_trace['showlegend'] = False
        traces.append( moon_trace)
        
        earth_curve = np.array(data.mkf['BR_EARTH'][i0:i1])
        earth_trace = go.Scatter(
                x=time,
                y=earth_curve,
                mode='markers',
                name='earth',
                marker=dict(color=green), 
                legendgroup='earth',
                xaxis=xnames[n],
                yaxis=ynames[2+n*6]
            )
        if n > 0: earth_trace['showlegend'] = False
        traces.append(earth_trace)

        pointing_curve = np.array(data.mkf['ANG_DIST'][i0:i1])
        pointing_trace = go.Scatter(
                x=time,
                y=pointing_curve,
                mode='markers',
                name='pointing',
                marker=dict(color=purple), 
                legendgroup='pointing',
                xaxis=xnames[n],
                yaxis=ynames[1+n*6]
            )
        if n > 0: pointing_trace['showlegend'] = False
        traces.append(pointing_trace)
        
        latitude_curve = np.array(data.mkf['SAT_LAT'][i0:i1])
        latitude_trace = go.Scatter(
                x=time,
                y=latitude_curve,
                mode='markers',
                name='latitude',
                marker=dict(color=blue), 
                legendgroup='latitude',
                xaxis=xnames[n],
                yaxis=ynames[0+n*6]
            )
        if n > 0: latitude_trace['showlegend'] = False
        traces.append(latitude_trace)


    # Construct the layout
    layout = go.Layout(
            height=850,
            legend=dict(orientation="h", x=-.1, y=1.2)
        )

    # Compute the grid ratio
    pti_fraction = ni.gtitools.durations(pti) / np.sum(ni.gtitools.durations(pti))
    if panel_mode == 'regular':
        pti_fraction = np.full(len(pti), 1.0/len(pti))
    grid_bounds = np.cumsum(np.concatenate(([0],pti_fraction)))
    x_start = np.copy(grid_bounds[:-1])
    x_stop  = np.copy(grid_bounds[1:])
    # Apply spacing
    spacing = 0.025
    x_start[1:] += spacing
    x_stop[:-1] -= spacing

    # Set horizontal layout
    for n,xax in enumerate(xnames):
        layout['xaxis{}'.format(n+1)]=dict(
                domain=[x_start[n], x_stop[n]],
                title='time <br> (+{:.0f} s)'.format(origin[n])
            )

    # Set the vertical layout
    for n,xax in enumerate(xnames):
        layout['yaxis{}'.format(6*n+1)]=dict(domain=[0.00, 0.15], anchor=xax)
        layout['yaxis{}'.format(6*n+2)]=dict(domain=[0.17, 0.32], anchor=xax, type='log')
        layout['yaxis{}'.format(6*n+3)]=dict(domain=[0.34, 0.49], anchor=xax)
        layout['yaxis{}'.format(6*n+4)]=dict(domain=[0.51, 0.66], anchor=xax)
        layout['yaxis{}'.format(6*n+5)]=dict(domain=[0.68, 0.83], anchor=xax)
        layout['yaxis{}'.format(6*n+6)]=dict(domain=[0.85, 1.00], anchor=xax)

    # Add ylabels
    layout['yaxis1'].update(title='angle (deg)')
    layout['yaxis2'].update(title='angle (deg)')
    layout['yaxis3'].update(title='angle (deg)')
    layout['yaxis4'].update(title='angle (deg)')
    layout['yaxis5'].update(title='ct/s')
    layout['yaxis6'].update(title='ct/s')
    
    return {
        'data': traces,
        'layout': layout
    }

