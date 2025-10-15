#! /usr/bin/env python3
# ^above is necessary to make executable
# use: chmod +x filename.py
from geometry import Apartment
import argparse
parser = argparse.ArgumentParser(description="Program for solving the stationaty heat equation in two dimensions",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-v", "--verbose", action="store_true",
                    help="increase output verbosity")
parser.add_argument("geometry",nargs="?", help="The geometry setting to be used. Alternatives: "
                    "\n  'default' : Uses the default 3-room layout"
                    "\n  'alternative' : Uses the alternative 4-room layout found in the appendix"
                    "\n  'custom' : Lets user define a custom geometry (interactive, CLI)", default="default", const="default")
parser.add_argument("-p", "--plot", action="store_true", help="plot the result")
parser.add_argument("-c", "--credits", action="store_true", help="show credits and exit")
parser.add_argument("-dx",help="define mesh sizing, defaults to 1/20:=0.05", default=1/20, type=float)
args=parser.parse_args()
if args.credits:
    print("Program written by: \n-Johan Fritz\n-Tobias Ryden\n-Samuel Eriksson\n-Valia Diamantaki &\n-Vahid Faraji\nas an assignment in the course 'Adva...")
    exit
if args.geometry=="custom":
    raise NotImplementedError("bruh")
apartment=Apartment(args.geometry, args.dx)
