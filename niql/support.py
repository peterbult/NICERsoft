import numpy as np
import pandas as pd
import nicerlab as ni

from astropy import table
from astropy.table import Table

def split_on_mpu(filename, clobber=False):
    return [
        ni.ftools.ftselect(
            inputfile=filename,
            outputfile="mpu{}_cl.evt".format(i),
            expression="(DET_ID>={}).and.(DET_ID<{})".format(i*10, (i+1)*10),
            clobber=clobber)
        for i in range(7)]

def filter_events(filenames, expression, clobber=False):
    return [
        ni.ftools.ftselect(
            inputfile=filename, 
            outputfile="filt_"+filename,
            expression=expression, 
            clobber=clobber)
        for filename in filenames]

#def generate_event_frame(filtnames):
    #all_data = [ni.io.read_from_fits(filtname, 
                                     #cols=['TIME', 'PI', 'DET_ID'], 
                                     #ext='EVENTS')
                #for filtname in filtnames]

    #frames = [pd.DataFrame(mpu, columns=['TIME', 'PI', 'DET']) for mpu in all_data]

    ## NOTE: FRAGILE: this can be broken if a MPU is missing
    #keys = ['mpu{}'.format(i) for i in range(len(filtnames))]

    #table = pd.concat(frames, axis=0, keys=keys)

    #return table

def generate_event_frame(table, cols=['TIME', 'PI', 'DET_ID']):
    mpus = table.group_by('MPU')

    frames = [
        pd.DataFrame( 
            np.array([mpu[c] for c in cols]).T,
            columns=cols
        ) for i,mpu in enumerate(mpus.groups)
    ]

    keys = ['mpu{}'.format(key._index) for key in mpus.groups.keys]

    return pd.concat(frames, axis=0, keys=keys)

def make_mpu_column(ufa):
    return Table.Column(divmod(ufa['DET_ID'], 10)[0], name='MPU')

