#! /usr/bin/env python3
# ^above is necessary to make executable
# use: chmod +x filename.py
from geometry import Apartment
from cli_parser import get_args


args = get_args()

if args.dx == None:
    dx = 1 / 20
else:
    dx = args.dx
if args.verbose:
    if args.dx == None:
        print("No value for 'dx' supplied, defaulting to dx:=0.05")
    else:
        print(f"Running script with dx={dx}")
    print(f"Apartment layout is set to '{args.geometry}'")

apartment = Apartment(args.geometry, dx)
if args.verbose:
    print("Successfully created apartment layout")
