import importlib.resources
from ctypes import CDLL
import sys

# Find shared library
assert (sys.platform == "linux") #only works with linux

_filename = importlib.resources.files(__name__)/f'lib/libomegah2csg.so'
_dll = CDLL(str(_filename))


from .config import *
#user need to call finalize

from .OmegaHMesh import *
from .openmcGeometry import *
