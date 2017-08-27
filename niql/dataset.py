import sys
from os import path
from glob import glob

import numpy as np
from astropy import log
from astropy.table import Table, vstack
from nicerlab import gtitools

class DataSet:
    """
    ObsID data loader for NICER. This class will scan 
    an ObsID directory and load relevant prefilter
    data.
    """
    def __init__(self, args):
        log.info('Initializing the ObsID data object')
        self.args = args

        # Verify that the directory is valid
        self._verify_directory(args.obsdir)

        # Read the event file
        self._read_evt(args.obsdir)

        # Read the prefilter data
        self._read_mkf(args.obsdir)

        # Read the gti table
        self._read_gti(args.obsdir)

        # Construct detector coordinate/name arrays
        self._prepare_detectors()


    def _read_evt(self, obsdir):
        """
        Read all *.evt files in the xti/event folder into memory.
        """
        # Grab the evt files
        self.evtfiles = glob(path.join(obsdir, 'xti/event_cl/ni*mpu?_cl.evt*'))
        
        # Verify
        if len(self.evtfiles) == 0:
            raise IOError('no event files found!')
        
        for f in self.evtfiles:
            if not path.isfile(f):
                raise IOError("'{}' is not a file")

        # Load the evt table
        self.evt = vstack([Table.read(evt, format='fits', hdu=1) for evt in self.evtfiles])


    def _read_mkf(self, obsdir):
        """
        Read the *.mkf filter file into memory.
        """
        # Grab the evt files
        self.mkffiles = glob(path.join(obsdir, 'auxil/ni*.mkf'))
        
        # Verify
        if len(self.mkffiles) == 0:
            raise IOError('no mkf file found!')

        for f in self.mkffiles:
            if not path.isfile(f):
                raise IOError("'{}' is not a file")

        # Load the mkf table
        self.mkf = vstack([Table.read(mkf, format='fits', hdu=1) for mkf in self.mkffiles])
        

    def _read_gti(self, obsdir):
        """
        Read all *.gti files in the xti/event folder into memory.
        """
        # Grab the gti files
        self.gtifiles = glob(path.join(obsdir, 'xti/event_cl/ni*mpu?_cl.evt*'))
        
        # Verify
        if len(self.gtifiles) == 0:
            raise IOError('no event files found!')
        
        for f in self.gtifiles:
            if not path.isfile(f):
                raise IOError("'{}' is not a file")

        # Load the gti table
        self.gti = gtitools.merge(
            [Table.read(gti, format='fits', hdu=2) for gti in self.gtifiles], 
            method='and'
        )


    def _verify_directory(self, obsdir):
        """
        Verify that a given path is a valid NICER ObsID.
        """
        # Remove any trailing slash
        obsdir = obsdir.rstrip('/')
        # Report
        log.info('Verifying directory \'{}\''.format(path.basename(obsdir)))
        # Specify required directories
        required_dirs = ['', 'auxil', 'xti']
        # Ensure all dirs exist
        for d in required_dirs:
            if not path.isdir(path.join(obsdir, d)):
                raise IOError("directory '{}' not found".format(d))


    def _prepare_detectors(self):
        det_names = []
        for mpu in range(7):
            for det in range(8):
                det_names.append(mpu*10+det)
        self.det_names = np.array(det_names)

        self.det_coords = [
            (6,0), (5,0), (4,0), (3,0), (2,0), (1,0), (0,0), (0,1),
            (6,1), (5,1), (4,1), (3,1), (2,1), (1,1), (0,2), (0,3),
            (6,2), (5,2), (4,2), (3,2), (2,2), (1,2), (1,3), (0,4),
            (6,3), (5,3), (4,3), (3,3), (2,3), (1,4), (1,5), (0,5),
            (6,4), (5,4), (4,4), (3,4), (2,4), (2,5), (1,6), (0,6),
            (6,5), (5,5), (4,5), (3,5), (2,6), (2,7), (1,7), (0,7),
            (6,6), (6,7), (5,6), (5,7), (4,6), (4,7), (3,6), (3,7)
        ]


