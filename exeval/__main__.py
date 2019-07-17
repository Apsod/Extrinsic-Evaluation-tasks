import argparse
from exeval import sequence_labeling, snli, subjectivity, relation_extraction, sentence_sentiment, document_sentiment
import logging
import os


MODULES = {
    'sequence_labeling': sequence_labeling,
    'snli': snli,
    'subjective': subjectivity,
    'relation': relation_extraction,
    'sentence_sentiment': sentence_sentiment,
    'document_sentiment': document_sentiment,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--backend', type=str)
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

    if args.backend:
        os.environ['KERAS_BACKEND'] = args.backend

    args.go(args)










