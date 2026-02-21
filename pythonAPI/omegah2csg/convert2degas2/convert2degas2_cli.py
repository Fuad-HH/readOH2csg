import argparse

from .convert2degas2 import convert2degas2


def main():
    parser = argparse.ArgumentParser(description='Convert Omega_h mesh to DEGAS2 geometry (netcdf file)')
    parser.add_argument('filename', help='Omega_h mesh file (.osh)')
    parser.add_argument('--tol', type=float, help='tolerance', default=1e-10)
    parser.add_argument('--auxfiles', type=bool, help='create dummy auxiliary files for degas2 case', default=False)
    args = parser.parse_args()

    convert2degas2(mesh_filename=args.filename, tol=args.tol, netcdf_filename="geometry.nc", create_aux_files=args.auxfiles)


# For setuptools entry point
app = main

if __name__ == '__main__':
    main()
