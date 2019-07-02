import argparse
from exeval import sequence_labeling, snli
import logging


MODULES = {
    'sequence_labeling': sequence_labeling,
    'snli': snli
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--vector_path',
                        required=True,
                        help='path to vectors (in text format)')


    subparser=parser.add_subparsers()

    for name, module in MODULES.items():
        sp = subparser.add_parser(name)
        module.mk_parser(sp)

    args = parser.parse_args()

    if args.log:
        logging.basicConfig(level=logging.DEBUG)

    args.go(args)










