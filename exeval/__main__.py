import argparse
from exeval import sequence_labeling
import logging


MODULES = {
    'sequence_labeling': sequence_labeling
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true')


    subparser=parser.add_subparsers()

    for name, module in MODULES.items():
        sp = subparser.add_parser(name)
        module.mk_parser(sp)

    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=logging.DEBUG)

    args.go(args)










