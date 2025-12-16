import argparse

from .convert2openmc import convert2openmc


def main():
    parser = argparse.ArgumentParser(description='Convert Omega_h mesh to OpenMC geometry')
    parser.add_argument('filename', help='Omega_h mesh file (.osh)')
    args = parser.parse_args()

    convert2openmc(args.filename)


# For setuptools entry point
app = main

if __name__ == '__main__':
    main()