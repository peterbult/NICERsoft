from __future__ import (print_function, division, unicode_literals, absolute_import)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from astropy import log

from nicer.plotutils import *
from nicer.fitsutils import *

def bkg_plots(etable, data, gtitable, args, mktable, shoottable):

    if args.eventshootrate:
        overshootrate = shoottable['EVENT_OVERSHOOTS']
        undershootrate = shoottable['EVENT_UNDERSHOOTS']
        bothrate = shoottable['EVENT_BOTHSHOOTS']
        hkmet = shoottable['HKMET']
    else:
        overshootrate = shoottable['HK_OVERSHOOTS']
        undershootrate = shoottable['HK_UNDERSHOOTS']
        bothrate = None
        hkmet = shoottable['HKMET']

    figure = plt.figure(figsize = (8.5, 11), facecolor = 'white')
    bkg_grid = gridspec.GridSpec(25,4)

    #Lightcurve of Rejected events
    log.info('Building Rejected Event Light curve')
    plt.subplot(bkg_grid[1:5,:4])

    hkmetbins = np.append(hkmet,(hkmet[-1]+hkmet[1]-hkmet[0]))

    # Extract bad ratio events and bin onto hkmet bins
    if len(data.ufafiles) == 0:
        badtable = get_badratioevents_ftools(data.evfiles,workdir=None)
    else:
        badtable = get_badratioevents_ftools(data.ufafiles,workdir=None)
    badlightcurve = np.histogram(badtable['TIME'], hkmetbins)[0]
    badlightcurve = np.array(badlightcurve,dtype=np.float)

    colornames = ['black','green','red','blue','magenta']
    colorlevels = np.arange(len(colornames))
    cmap, norm = mpl.colors.from_levels_and_colors(levels=colorlevels, colors=colornames, extend='max')

    times, lc, cc = convert_to_elapsed_goodtime(hkmet, badlightcurve, gtitable)
    plot.scatter(times, lc, c=np.fmod(cc,len(colornames)), cmap=cmap, norm=norm, marker='.')

    # Really should convolve in GTI segments!
    kernel = np.ones(32)/32.0
    badsmooth = np.convolve(badlightcurve,kernel,mode='same')
    times, lc, cc = convert_to_elapsed_goodtime(hkmet, badsmooth, gtitable)
    plot.plot(times, lc)

    plot.yscale('symlog',linthreshy=5.0)
    plot.ylim(ymax=100.0)
    #plt.legend(handles = [lc], loc = 2)
    plot.grid(True)
    plt.annotate('Ratio-rejected event light curve', xy=(0.03, 0.85), xycoords='axes fraction')

    #Overshoot rate plot -- use --lclog to make it a log y axis
    log.info('Building overshoot plot')
    plt.subplot(bkg_grid[5:9,:4])
    if overshootrate is not None:
        plot_overshoot(etable, overshootrate, gtitable, args, hkmet, bothrate)
        plot_SAA(mktable, gtitable, overshootrate)

    #Plot of undershoot rate
    log.info('Building undershoot plot')
    plt.subplot(bkg_grid[9:13,:4])
    if undershootrate is not None:
        plot_undershoot(etable, undershootrate, gtitable, args, hkmet, mktable)

    #Plot of Sun / Moon
    log.info('Building Sun / Moon / Earth angle Plot')
    plt.subplot(bkg_grid[13:17,:4])
    plot_angles(mktable, gtitable)

    #Pointing plot
    plt.subplot(bkg_grid[17:21,:4])
    plot_pointing(mktable, gtitable)

    #Plot of LAT / Long
    plt.subplot(bkg_grid[21:25,:4])
    plot_latlon(mktable, gtitable)


    figure.suptitle('Object: {0} at {1}'.format(etable.meta['OBJECT'],etable.meta['DATE-OBS'].replace('T', ' at ')),
                    fontsize=16)
    #plt.subplots_adjust(left = .07, right = .99, bottom = .05, top = .9, wspace = .95, hspace = .95)
    plt.tight_layout()
    return figure
