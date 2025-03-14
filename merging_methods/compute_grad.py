import sys
sys.path.append('.')

import re
import os
import logging
import torch
import torch.nn as nn
import numpy as np

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from datasets import Dataset, concatenate_datasets, load_from_disk
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from multiprocessing import Pool


from utils.utils import load_json


def dataset_split(dataset, num):
    chunk_size = len(dataset)//num
    return [dataset[i:i+chunk_size] for i in range(0,len(dataset),chunk_size)]


def compute_grad(dataset, device_parallel_size, model, tokenizer, infer_batch_size, max_seq_length, chat_template, outputs_dir='./results', save=True):
    
    # mp.set_start_method("spawn")

    dataset_slices = dataset_split(dataset, device_parallel_size)
    
    # prepare args
    args_for_devices = []
    for idx in range(device_parallel_size):
        inter_dataset_slice = dataset_slices[idx]
        device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
        grad_path = outputs_dir + '/' + 'grad_no_pad_token_cuda' + str(idx) + '.pth'
        args_for_devices.append((inter_dataset_slice, infer_batch_size, max_seq_length, device, model, tokenizer, chat_template, grad_path))
    
    # parallel inference
    with Pool(device_parallel_size, maxtasksperchild=1) as p:
        p.map(compute_grad_on_device, args_for_devices)

    print('Compute grad done \nStart Merging')
    grad_slice = []
    for idx in range(device_parallel_size):
        grad_path = outputs_dir + '/' + 'grad_no_pad_token_cuda' + str(idx) + '.pth'
        grad_slice.append(torch.load(grad_path,weights_only=True))
    
    merged_grad = {}
    for key, value in grad_slice[0].items():
        merged_grad[key] = torch.sum(torch.stack([gs[key] for gs in grad_slice]),dim=0).to(torch.bfloat16)
        # import pdb;pdb.set_trace()

    if save == True:
        merged_grad_path = outputs_dir + '/' + 'merged_grad_no_pad_token.pth'
        torch.save(merged_grad,merged_grad_path)
    
    print('Done')

    return merged_grad


def compute_grad_on_device(args):
    inter_dataset_slice, infer_batch_size, max_seq_length, device, model, tokenizer, chat_template, path = args
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype="bfloat16")
    model = model.to(device)
    # model.eval()

    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    if tokenizer.chat_template == None:
        tokenizer.chat_template = chat_template
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # inter_dataset_slice = inter_dataset_slice.map(map_chat_template, fn_kwargs={"tokenizer": tokenizer}, num_proc=1)

    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    for i in tqdm(range(0, len(inter_dataset_slice), infer_batch_size), desc="Computing grad"):
        inputs = tokenizer.apply_chat_template(
            inter_dataset_slice[i:i + infer_batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
            return_dict=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()} # type: ignore
        # mask = inputs["attention_mask"].clone()
        labels = inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding

        outputs = model(**inputs)
        logits = outputs["logits"]

        # Shift logits and labels to calculate loss for each token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # bs, seq_len = shift_labels.shape
        loss = loss_fct(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
        loss = torch.sum(loss)
        loss.backward()

    # get grad
    grad = {}
    for key, value in model.named_parameters():
        grad[key] = value.grad.cpu()
    
    torch.save(grad, path)
    # print('good')
    # return grad


# if __name__ == "__main__":

#     mp.set_start_method("spawn")

#     data_json = load_json('datasets/gsm8k_calibration_chat.json')
#     # print(len(data_json))
#     # data_dict = {'ct':data_json}
#     tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-7B-Instruct')
#     chat_template = tokenizer.chat_template

#     compute_grad(data_json, 4, 'Qwen/Qwen2.5-7B', 4, 1024, chat_template)

