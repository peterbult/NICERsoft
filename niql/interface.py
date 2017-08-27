import os
import sys
import argparse
import numpy as np
from astropy import log

from dataset import DataSet
from support import make_mpu_column, generate_event_frame
from server import cache

@cache.memoize(timeout=3600)  # in seconds
def cached_FileSet(args):
    return DataSet(args)

# Parse from the command line
parser = argparse.ArgumentParser(description="niql: a NICER quicklook tool")
parser.add_argument("--obsdir", help="Give ObsID root")
parser.add_argument("--object", help="Override object name", default=None)
args = parser.parse_args()

# Ensure the directory is valid
if (args.obsdir is None):
    sys.exit("> give obsdir")

if not os.path.isdir(args.obsdir):
    sys.exit("> invalid directory")

try:
    # Load the nicer data
    data = cached_FileSet(args)
    # Add a MPU column to the data
    data.evt.add_column(make_mpu_column(data.evt))
except Exception as e:
    log.error(e)
    exit(1)

