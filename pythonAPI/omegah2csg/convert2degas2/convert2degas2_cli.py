import argparse

from .. import __version__
from .convert2degas2 import convert2degas2


def main():
    parser = argparse.ArgumentParser(
        description="Convert Omega_h mesh to DEGAS2 geometry (netcdf file)",
        epilog="Example usage: %(prog)s  mesh.osh --tol 1e-8 --auxfiles --target-temperature 300 --target-recyc-coef 0.5",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "filename",
        help="Omega_h mesh file (.osh). The mesh must have a boundary rectangle defined with tag 'offset_face' and wall nodes as 'isOnWall'. Check the documentation for details.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Tolerance for geometric comparisons (default: 1e-10). Try smaller values if you encounter issues with finer meshes.",
    )
    parser.add_argument(
        "--auxfiles",
        "-a",
        action="store_true",
        help="Create auxiliary files (plasmafile.txt, sourcefile.txt, wallfile.txt) required for Degas2 simulations",
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=300.0,
        help="Target temperature in Kelvin (default: 300 K)",
    )
    parser.add_argument(
        "--output-filename",
        "-o",
        type=str,
        default="geometry.nc",
        help="Output netCDF filename (default: geometry.nc)",
    )
    parser.add_argument(
        "--target-recyc-coef",
        type=float,
        default=0.50,
        help="Target recycling coefficient (default: 0.50)",
    )
    args = parser.parse_args()

    convert2degas2(
        mesh_filename=args.filename,
        tol=args.tol,
        netcdf_filename=args.output_filename,
        create_aux_files=args.auxfiles,
        target_temperature_K=args.target_temperature,
        target_recyc_coef=args.target_recyc_coef,
    )


# For setuptools entry point
app = main

if __name__ == "__main__":
    main()
