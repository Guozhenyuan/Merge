
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse
import torch.nn as nn
import random
from datasets import Dataset
from torch.utils.data import DataLoader

class WrappedGPT:
    """ 收集每个线性层的输入激活的L2范数"""
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name
    
    def add_batch(self, inp, out):
        """累积计算输入激活的统计量"""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, torch.nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        
        # 更新统计量使用加权平均
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def get_loaders(name, nsamples=128, seed=0, seqlen=512, tokenizer=None):
    if "gsm8k" in name:
        return get_gsm8k(nsamples, seed, seqlen, tokenizer)
    
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_gsm8k(nsamples, seed, seqlen=512, tokenizer=None):
    # Load train and validation datasets
    # traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    with open("/zju_wck/czc/dataset/gsm8k_calibration_chat.json", 'r', encoding='utf-8') as f:
                valdata = json.load(f)
    # Generate samples from training set
    traindata=valdata
    random.seed(seed)
    trainloader = []
    # import pdb ; pdb.set_trace()
    for _ in range(nsamples):#默认裁切128条
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer('\n'.join(f"{item['role']}: {item['content']}" for item in traindata[i]), return_tensors='pt')
            # tokenizer.apply_chat_template(calibration_data,padding=True,max_length=1024,return_dict=True,return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    # valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    # valenc = valenc.input_ids[:, :(256 * seqlen)]
    # valenc = TokenizerWrapper(valenc)
    valenc = tokenizer.apply_chat_template(
        valdata[:1100],  # 取前 1100 个样本
        padding=True,
        max_length=seqlen, 
        return_dict=True,
        return_tensors='pt'
    )
    valenc['labels'] = valenc['input_ids'].clone()
    # 为了兼容 TokenizerWrapper，只保留 input_ids
    valenc = TokenizerWrapper(valenc['input_ids'])
    # import pdb ; pdb.set_trace()
    return trainloader, valenc


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype#获取类型
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)#创建一个张量
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}#字典保存数据

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    # import pdb; pdb.set_trace()
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    # import pdb; pdb.set_trace()
    return inps, outs, attention_mask, position_ids 


def calculate_wanda_importance(model, tokenizer, calibration_file, n_samples=128, seqlength=512,sparsity_ratio=0.5, device="cuda"):
    """s
    计算模型参数的Wanda重要性度量和掩码
    参数:
        model
        tokenizer
        calibration_file: 校准数据集文件路径
        n_samples: 使用的校准样本数量
        sparsity_ratio: 目标稀疏率
        device: 计算设备
        use_variant: 是否使用Wanda变体
        
    返回:
        layer_masks: 每层的W_mask字典
    """
    print(f"Loading model to {device}")
    model.seqlen = seqlength
    model = model.to(device)
    model.eval()
    
    # 关闭缓存以确保正确的前向传播
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    print(f"Loading calibration data from {calibration_file}")
    # encodings = load_calibration_dataset(calibration_file, tokenizer, n_samples)
    dataloader, _ = get_loaders("gsm8k",nsamples=n_samples,seed=0,seqlen=model.seqlen,tokenizer=tokenizer)
    print("Preparing calibration inputs")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
  
    # 存储每层的mertics
    layer_mertics = {}
    
    # 遍历模型的每一层
    layers = model.model.layers
    print(f"Processing {len(layers)} layers")
    
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = find_layers(layer)
        
        # 处理多GPU情况
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        # import pdb; pdb.set_trace()
        
        # 初始化包装层
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
        
        # 注册前向hook
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        
        for j in range(n_samples):#每条数据
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]#收集数据

        # 移除hook
        for h in handles:
            h.remove()

        # import pdb; pdb.set_trace()
        layer_mertics[i] = {}
        for name in subset:
            print(f"Computing importance for layer {i}, module {name}")
            # 计算重要性度量
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
            print(f"Layer {i}, module {name}: sparsity={sparsity_ratio}")
            layer_mertics[i][name] = W_metric


        for j in range(n_samples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps
    
    # 恢复模型原始配置
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    
    return layer_mertics



