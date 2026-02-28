import importlib.resources
import sys
from ctypes import CDLL

assert sys.platform == "linux"  # only works with linux

_filename = importlib.resources.files(__package__) / "lib/libomegah2csg.so"
_dll = CDLL(str(_filename))
