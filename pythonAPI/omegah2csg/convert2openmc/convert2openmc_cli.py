import argparse

from .. import __version__
from .convert2openmc import convert2openmcXML


def main():
    parser = argparse.ArgumentParser(
        description="Convert Omega_h mesh to OpenMC geometry"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument("filename", help="Omega_h mesh file (.osh)")
    parser.add_argument("--tol", type=float, help="tolerance", default=1e-10)
    args = parser.parse_args()

    convert2openmcXML(filename=args.filename, tol=args.tol)


# For setuptools entry point
app = main

if __name__ == "__main__":
    main()
