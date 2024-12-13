import logging
import torch
import random
import numpy as np
from utility.log_helper import *

import json
import torch.optim as optim

from time import time
from typing import Callable, Dict, List, Optional, Type, Union, Tuple
from utility.parser_seed import parse_args
from model.DearLLM import DearLLM
from datasets import SampleBaseDataset
from sklearn.model_selection import train_test_split
from loader import get_dataloader

from metrics import get_metrics_fn
import os
import pickle

def patient_train_val_test_split(
        dataset: SampleBaseDataset,
        ratios: Union[Tuple[float, float, float], List[float]],
        seed: Optional[int] = None,
):
    if seed is not None:
        np.random.seed(seed)
    assert sum(ratios) == 1.0, "ratios must sum to 1.0"
    patient_indx = list(range(0, len(dataset), 1))
    label_list = [sample["label"] for sample in dataset]
    temp_index, test_index, y_temp, y_test = \
        train_test_split(patient_indx, label_list, test_size=ratios[2], stratify=label_list, random_state=seed)
    train_index, val_index, y_train, y_val = train_test_split(temp_index, y_temp,
                                                              test_size=ratios[1] / ratios[0] + ratios[1],
                                                              stratify=y_temp,
                                                              random_state=seed)

    train_dataset = torch.utils.data.Subset(dataset, train_index)
    val_dataset = torch.utils.data.Subset(dataset, val_index)
    test_dataset = torch.utils.data.Subset(dataset, test_index)
    return train_dataset, val_dataset, test_dataset

def is_best(best_score: float, score: float, monitor_criterion: str) -> bool:
    if monitor_criterion == "max":
        return score > best_score
    elif monitor_criterion == "min":
        return score < best_score
    else:
        raise ValueError(f"Monitor criterion {monitor_criterion} is not supported")

def inference(model, dataloader, additional_outputs=None) -> Dict[str, float]:
    loss_all = []
    y_true_all = []
    y_prob_all = []
    if additional_outputs is not None:
        additional_outputs = {k: [] for k in additional_outputs}
    for data, graph_batch in dataloader:
        model.eval()
        with torch.no_grad():
            graph_batch.to(model.device)
            loss_cls, cls_logits, patient_y_true, patient_y_prob, patient_embed = model(
                graph_batch, **data)
            y_true = patient_y_true.data.cpu().numpy()
            y_prob = patient_y_prob.data.cpu().numpy()
            loss_all.append(loss_cls.item())
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
    loss_mean = sum(loss_all) / len(loss_all)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)
    if additional_outputs is not None:
        additional_outputs = {key: np.concatenate(val)
                              for key, val in additional_outputs.items()}
        return y_true_all, y_prob_all, loss_mean, additional_outputs
    return y_true_all, y_prob_all, loss_mean

def evaluate(model, dataloader, metrics) -> Dict[str, float]:
    y_true_all, y_prob_all, loss_mean = inference(model, dataloader)
    mode = model.model.mode
    metrics_fn = get_metrics_fn(mode)
    scores = metrics_fn(y_true_all, y_prob_all, metrics=metrics)
    return scores

def main(args, singlerun_seed):
    random.seed(singlerun_seed)
    np.random.seed(singlerun_seed)
    torch.manual_seed(singlerun_seed)

    save_dir = args.save_dir + 'trlr{}_wdcay{}_model{}_hdim{}_edim{}_elayer{}_nodedim{}_gdim-{}_seed{}_{}/'.format(
        args.train_lr,
        args.weight_decay,
        args.modeltype,
        args.hidden_dim,
        args.embed_dim,
        args.encoder_layer,
        args.node_dim,
        args.gencoder_dim_list,
        singlerun_seed,
        args.cuda_choice)

    log_save_id = create_log_id(save_dir)
    logging_config(folder=save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    def load_pickle(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    if args.dataset == "MIMIC3":
        datasample, datagraph_sample = load_pickle("./mimic3.pkl")
        x_key = ["conditions"]
        id_to_icd_map_path = "./id_to_icd.json"
        with open(id_to_icd_map_path, "r") as f:
            id_to_icd_map = json.load(f)

    elif args.dataset == "MIMIC4":
        datasample, datagraph_sample = load_pickle("./mimic4.pkl")
        x_key = ["conditions"]
        id_to_icd_map_path = "./id_to_icd.json"
        with open(id_to_icd_map_path, "r") as f:
            id_to_icd_map = json.load(f)

    node_num = len(id_to_icd_map)

    use_cuda = torch.cuda.is_available()
    device = torch.device(args.cuda_choice if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(singlerun_seed)
    torch.backends.cudnn.deterministic = True

    train_ds, val_ds, test_ds = patient_train_val_test_split(datagraph_sample, eval(args.train_val_test_split),
                                                             singlerun_seed)

    train_dataloader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dataloader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_dataloader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)

    model_kwargs = {"dataset": datasample, "feature_keys": x_key, "label_key": "label", "mode": "binary", "node_num": node_num}
    model = DearLLM(args, **model_kwargs)
    model.to(device)
    logging.info(model)
    logging.info(args.cuda_choice)
    with open(save_dir + "params.json", mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    last_checkpoint_path_name = os.path.join(save_dir, "last.ckpt")
    best_checkpoint_path_name = os.path.join(save_dir, "best.ckpt")

    if args.use_last_checkpoint != -1:
        logging.info(f"Loading checkpoint from {last_checkpoint_path_name}")
        state_dict = torch.load(last_checkpoint_path_name, map_location=args.cuda_choice)
        model.load_state_dict(state_dict)
    logging.info("")

    logging.info("Training:")
    param = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param if (not any(nd in n for nd in no_decay)) and (not "gencoders" in n)],
            "lr": args.train_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in param if (not any(nd in n for nd in no_decay)) and ("gencoders" in n)],
            "lr": args.gencoder_lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in param if (any(nd in n for nd in no_decay)) and (not "gencoders" in n)],
            "lr": args.train_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in param if (any(nd in n for nd in no_decay)) and ("gencoders" in n)],
            "lr": args.gencoder_lr,
            "weight_decay": 0.0,
        },
    ]

    optimizer_train = optim.Adam(optimizer_grouped_parameters)
    data_iterator = iter(train_dataloader)
    best_score = -1 * float("inf") if args.monitor_criterion == "max" else float("inf")
    steps_per_epoch = len(train_dataloader)
    global_step = 0
    best_dev_epoch = 0

    for epoch in range(args.epochs_train):
        time0 = time()
        training_loss_all = []
        model.train()

        for _ in range(steps_per_epoch):
            optimizer_train.zero_grad()
            try:
                data,graph_batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(train_dataloader)
                data,graph_batch = next(data_iterator)
            graph_batch.to(device)
            loss_cls, cls_logits, patient_y_true, patient_y_prob, patient_embed = model(
                graph_batch, **data)
            loss_cls.backward()

            if args.clip != -1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer_train.step()

            training_loss_all.append(loss_cls.item())
            global_step += 1

        logging.info("--- Train epoch-{}, step-{}, Total Time {:.1f}s---".format(epoch, global_step, time() - time0))
        logging.info(f"loss: {sum(training_loss_all) / len(training_loss_all):.4f}")

        if last_checkpoint_path_name is not None:
            state_dict = model.state_dict()
            torch.save(state_dict, last_checkpoint_path_name)

        if val_dataloader is not None:
            scores = evaluate(model, val_dataloader, args.metrics)
            logging.info(f"--- Eval epoch-{epoch}, step-{global_step} ---")
            logging.info(f"--- Val Metrics ---")
            for key in scores.keys():
                logging.info("{}: {:.4f}".format(key, scores[key]))
            if args.monitor is not None:
                assert args.monitor in args.metrics, "monitor not in metrics!"
                score = scores[args.monitor]
                if is_best(best_score, score, args.monitor_criterion):
                    logging.info(
                        f"New best {args.monitor} score ({score:.4f}) "
                        f"at epoch-{epoch}, step-{global_step}"
                    )
                    best_dev_epoch = epoch
                    best_score = score
                    if best_checkpoint_path_name is not None:
                        state_dict = model.state_dict()
                        torch.save(state_dict, best_checkpoint_path_name)

        if epoch > args.unfreeze_epoch and epoch - best_dev_epoch >= args.max_epochs_before_stop:
            break

    logging.info('Best eval score: {:.4f} (at epoch {})'.format(best_score, best_dev_epoch))

    if os.path.isfile(best_checkpoint_path_name):
        logging.info("Loaded best model")
        state_dict = torch.load(best_checkpoint_path_name, map_location=args.cuda_choice)
        model.load_state_dict(state_dict)
    if test_dataloader is not None:
        scores = evaluate(model, test_dataloader, args.metrics)
        logging.info(f"--- Test ---")
        for key in scores.keys():
            logging.info("{}: {:.4f}".format(key, scores[key]))

    return scores


if __name__ == "__main__":
    args = parse_args()
    all_scores = []
    for seed in args.seed:
        scores = main(args,seed)
        all_scores.append(scores)
    print(all_scores)
    for metric in args.metrics:
        print("{}:{:.4f}({:.4f})".format(metric,np.mean([score[metric] for score in all_scores]),
                                         np.std([score[metric] for score in all_scores])))