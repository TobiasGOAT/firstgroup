#! /usr/bin/env python3
# ^above is necessary to make executable
# use: chmod +x filename.py
from geometry import Apartment
from cli_parser import args
from cli_helpers import *

if args.verbose:
    print(f"Apartment layout is set to '{args.geometry}'")

apartment = Apartment(args.geometry, args.dx)
if args.verbose:
    print(
        f"Successfully created apartment layout consisting of {len(apartment.rooms)} rooms"
    )
for i in range(args.iterations):
    if args.verbose:
        print(bold(f"Iteration {i}:"))
    apartment.iterate()
if args.plot:
    apartment.plot()
