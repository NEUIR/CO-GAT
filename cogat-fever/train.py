import sys
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AutoModel
)
import transformers
import os
from os import listdir
from os.path import isfile, join
import argparse
import glob
import json
import logging
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, DistributedSampler
from tqdm import tqdm, trange
import torch.distributed as dist
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_loader import data_loader
from models import COGATModel

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def get_metric(preds, golds):
    golds = [gold for gold in golds]
    preds = [pred for pred in preds]
    acc = accuracy_score(golds, preds)
    f1_macro = f1_score(golds, preds, average='macro')
    return {'acc': acc, 'f1': f1_macro}




def eval_model(model, valid_dataloader):
    model.eval()
    predict_list = []
    golden_list = []
    with torch.no_grad():
        for step, batch in enumerate(valid_dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            labels = batch["labels"]
            if 'token_type_ids' in batch:
                token_type_ids = batch["token_type_ids"].cuda()
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                outputs = model(input_ids, attention_mask)
            max_score, max_idxs = torch.max(outputs, 1)
            predict_idxs = max_idxs.view(-1).tolist()
            predict_list.extend(predict_idxs)
            golden_idxs = labels.view(-1).tolist()
            golden_list.extend(golden_idxs)
        evaluation_results = get_metric(predict_list, golden_list)
        return evaluation_results


def train(args, model, train_dataloader, valid_dataloader):
    writer = SummaryWriter(log_dir=args.output_dir)
    """ Train the model """

    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps
    t_total = train_dataloader.dataset.__len__() // real_batch_size * args.num_train_epochs
    if args.pre_attention:
        trainable_parameters = []
        trainable_names = []
        trainable_components = ['attentions']
        for name, param in model.named_parameters():
            param.requires_grad = False
            for component in trainable_components:
                if component in name:
                    trainable_parameters.append(param)
                    trainable_names.append(name)
                    param.requires_grad = True
                    break
        logger.info("*** pre_attention Training *****")
        logger.info(trainable_names)
        optimizer_grouped_parameters = [
            {'params': trainable_parameters, 'weight_decay': 0.0}
        ]
    else:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    patience_counter = 0
    global_step = 0
    # Check if continuing training from a checkpoint
    tr_loss = 0.0
    best_acc = 0.0
    model.zero_grad()

    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            model.train()
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()

            labels = batch["labels"].cuda()
            ground_truth = batch["ground_truth"].cuda()
            if args.roberta:
                loss = model(input_ids, attention_mask, ground_truth=ground_truth,
                             labels=labels)
            else:
                token_type_ids = batch["token_type_ids"].cuda()
                loss = model(input_ids, attention_mask, token_type_ids, ground_truth=ground_truth,
                         labels=labels)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (tr_loss / global_step)))
                writer.add_scalar("train_loss", tr_loss / global_step, global_step)  # 在每次计算完 loss 后添加
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                if global_step % args.eval_steps == 0:
                    logger.info('Start eval!')
                    evaluation_results = eval_model(model, valid_dataloader)
                    acc = evaluation_results["acc"]
                    f1 = evaluation_results["f1"]
                    logger.info('Dev acc: {0}, F1: {1}'.format(acc, f1))
                    writer.add_scalar("val_acc", acc, epoch)
                    if f1 > best_acc:
                        best_acc = f1
                        patience_counter = 0
                        torch.save({'epoch': epoch,
                                    'model': model.state_dict()}, os.path.join(args.output_dir, "save_model.best.pt"))
                        logger.info("Saved best epoch {0}, best F1 {1}".format(epoch, best_acc))
                    else:
                        patience_counter += 1
            if patience_counter > args.patience:
                print("Early stopping...")
                break
        if patience_counter > args.patience:
            print("Early stopping...")
            break
    writer.close()

def load_stuff(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = COGATModel(args, tokenizer)
    if args.checkpoint != None:
        model.load_state_dict(torch.load(args.checkpoint)['model'], strict=False)
    model.cuda()
    return tokenizer, model

def get_arguments():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument(
        "--train_data_path",
        default=None,
        type=str,
        required=True,
        help="The training data path.",
    )

    parser.add_argument(
        "--valid_data_path",
        default=None,
        type=str,
        required=True,
        help="The validation data path.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        # required=True,
        help="The checkpoint path.",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="The model path",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )


    parser.add_argument(
        "--max_len",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization.",
    )


    parser.add_argument(
        "--train_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--valid_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.",
    )

    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=5.0,
        type=float,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.",
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
        "--evi_num",
        type=int,
        default=5,
        help='Evidence num.',
    )
    parser.add_argument('--patience', type=int, default=5, help='Patience')

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="eval model every X updates steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed for initialization",
    )
    parser.add_argument(
        "--pre_attention",
        default=False,
        action="store_true",
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
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    handlers = [logging.FileHandler(os.path.abspath(args.output_dir) + '/train_log.txt'), logging.StreamHandler()]
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

    logger.info("Loading training set.")
    train_dataset = data_loader(args, args.train_data_path, tokenizer)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=train_dataset.collect_fn
    )

    logger.info("Loading validation set.")
    valid_dataset = data_loader(args, args.valid_data_path, tokenizer)
    sampler = SequentialSampler(valid_dataset)

    valid_dataloader = DataLoader(
        valid_dataset,
        sampler=sampler,
        batch_size=args.valid_batch_size,
        drop_last=False,
        num_workers=10,
        collate_fn=valid_dataset.collect_fn)


    # Training
    ##load dpt model

    train(args, model, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()