import argparse
import json

from lib.data import GoldDataset, PredictedDataset
from lib import metrics


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate SciFact predictions.'
    )
    parser.add_argument('--gold', type=str, required=True,
                        help='The gold labels.')
    parser.add_argument('--corpus', type=str, required=True,
                        help='The corpus of documents.')
    parser.add_argument('--prediction', type=str, required=True,
                        help='The predictions.')
    # parser.add_argument('--rationale-selection', type=str, required=True)
    parser.add_argument('--output', type=str, default=None,
                        help='If provided, save metrics to this file.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    data = GoldDataset(args.corpus, args.gold)
    predictions = PredictedDataset(data, args.prediction)
    # predictions = PredictedDataset(
    #     data,
    #     args.rationale_selection,
    #     args.prediction
    # )
    res = metrics.compute_metrics(predictions)
    print(res)
    # if args.output is not None:
    #     with open(args.output, "w") as f:
    #         json.dump(res.to_dict(), f, indent=2)
    # else:
    #     print(res)


if __name__ == "__main__":
    main()