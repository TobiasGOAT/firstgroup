import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="Program for solving the stationaty heat equation in two dimensions",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity"
    )
    parser.add_argument(
        "geometry",
        nargs="?",
        help="The geometry setting to be used. Alternatives: "
        "\n  'default' : Uses the default 3-room layout"
        "\n  'alternative' : Uses the alternative 4-room layout found in the appendix"
        "\n  'custom' : Lets user define a custom geometry (interactive, CLI)",
        default="default",
        const="default",
    )
    parser.add_argument("-p", "--plot", action="store_true", help="plot the result")
    parser.add_argument(
        "-c", "--credits", action="store_true", help="show credits and exit"
    )
    parser.add_argument(
        "-dx",
        "--dx",
        help="define mesh sizing, defaults to 1/20:=0.05",
        default=None,
        type=float,
    )

    args = parser.parse_args()
    if args.credits:
        authors = [
            "Johan Fritz",
            "Tobias Ryden",
            "Samuel Eriksson",
            "Valia Diamantaki",
            "Vahid Faraji",
        ]
        print("Program written by:")
        for a in authors:
            print(f"    -{a}")
        print("as a project in the course")
        print("    'Advanced Course in Numerical Algorithms with Python/SciPy'")
        print("at Lund University")
        exit
    return args
