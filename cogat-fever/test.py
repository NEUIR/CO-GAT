import json

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModel, ElectraTokenizer
)
from models import COGATModel
import argparse
import logging
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
from data_loader import data_loader_test
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def eval_model(model, valid_dataloader,outpath):
    model.eval()
    predict_list = []
    ids_list = []
    label_list = ["SUPPORTS", "REFUTES","NOT ENOUGH INFO"]
    with torch.no_grad():
        for step, batch in enumerate(valid_dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            if 'token_type_ids' in batch:
                token_type_ids = batch["token_type_ids"].cuda()
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                outputs = model(input_ids, attention_mask)
            ids = batch['ids']
            max_score, max_idxs = torch.max(outputs, 1)
            predict_idxs = max_idxs.view(-1).tolist()
            predict_list.extend(predict_idxs)
            idxs = ids.view(-1).tolist()
            ids_list.extend(idxs)
        with open(outpath, "w") as f:
            for step in range(len(predict_list)):
                instance = {"id": ids_list[step], "predicted_label": label_list[predict_list[step]]}
                f.write(json.dumps(instance) + "\n")


def load_stuff(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = COGATModel(args, tokenizer)
    if args.checkpoint != None:
        model.load_state_dict(torch.load(args.checkpoint)['model'])
    model.cuda()
    return tokenizer, model



def get_arguments():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(
        "--valid_data_path",
        default=None,
        type=str,
        required=True,
        help="The validation data path.",
    )
    parser.add_argument(
        "--outputpath",
        default=None,
        type=str,
        required=True,
        help="The output path of the predict result.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        # required=True,
        help="The checkpoint data path.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max_len",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )
    parser.add_argument(
        "--valid_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--hidden_size",
        default=768,
        type=int,
        help="Hidden size.",
    )

    parser.add_argument(
        "--project_dim",
        default=3,
        type=int,
        help="Project Dim.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed for initialization",
    )

    parser.add_argument(
        "--evi_num",
        type=int,
        default=5,
        help='Evidence num.',
    )
    parser.add_argument(
        "--ablation",
        default=False,
        action="store_true",
        help="use the only fact loss as the final loss function",
    )
    parser.add_argument(
        "--roberta",
        default=False,
        action="store_true",
        help="use the roberta model",
    )
    args = parser.parse_args()

    return args


def set_env(args):
    handlers = [logging.FileHandler('./evaluation_log.txt'), logging.StreamHandler()]
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers = handlers)
    # Set seed
    set_seed(args)


def main():
    args = get_arguments()
    set_env(args)
    logger.info("Training/evaluation parameters %s", args)
    tokenizer, model = load_stuff(args)

    logger.info("Loading validation set.")
    valid_dataset = data_loader_test(args, args.valid_data_path, tokenizer)
    sampler = SequentialSampler(valid_dataset)

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=valid_dataset.collect_fn)


    # Training

    eval_model(model, valid_dataloader, args.outputpath)


if __name__ == "__main__":
    main()