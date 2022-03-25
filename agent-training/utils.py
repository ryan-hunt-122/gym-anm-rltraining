import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str)
    parser.add_argument("--path", "-p", type=str, default=None)

    args = parser.parse_args()

    return args