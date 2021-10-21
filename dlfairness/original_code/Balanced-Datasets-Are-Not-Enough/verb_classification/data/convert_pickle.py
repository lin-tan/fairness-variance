import os
import dill
import pickle
import argparse
import shutil


def convert(old_pkl):
    """
    Convert a Python 2 pickle to Python 3
    """
    # Make a name for the new pickle
    new_pkl = os.path.basename(old_pkl)
    old_plk_rename = old_pkl + ".p2.bak"

    # Convert Python 2 "ObjectType" to Python 3 object
    dill._dill._reverse_typemap["ObjectType"] = object

    # Open the pickle using latin1 encoding
    with open(old_pkl, "rb") as f:
        loaded = pickle.load(f, encoding="latin1")

    #print(loaded)

    shutil.move(old_pkl, old_plk_rename)

    # Re-save as Python 3 pickle
    with open(new_pkl, "wb") as outfile:
        pickle.dump(loaded, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Python 2 pickle to Python 3"
    )

    parser.add_argument("infile", help="Python 2 pickle filename")

    args = parser.parse_args()

    convert(args.infile)