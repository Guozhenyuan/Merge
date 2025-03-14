import os
import torch
import argparse
from typing import List, Dict, Tuple
from transformers import PreTrainedModel,PreTrainedTokenizer

class MergeBase:
    def __init__(self, 
                 args,
                 pretrained_model: PreTrainedModel, 
                 finetuned_models: List[PreTrainedModel]):
        
        self.device = args.device
        self.merged_model = None
        self.output_dir = args.output

        if pretrained_model == None:
            self.pretrained_model = finetuned_models[0]
        else:
            self.pretrained_model = pretrained_model
        self.finetuned_models = finetuned_models

    def process_finetuned_models(self)->Dict[str,List[torch.Tensor]]:
        finetuned_models_dict = {}
        for name_param, tensor in self.pretrained_model.named_parameters():
            finetuned_models_dict[name_param] = [ftm.state_dict()[name_param] for ftm in self.finetuned_models]
        return finetuned_models_dict

    def get_task_vector(self,finetuned_models_dict:Dict[str,List[torch.Tensor]])->Dict[str,List[torch.Tensor]]:
        task_vector = {}
        for name_param, tensor in self.pretrained_model.named_parameters():
            pt_tensor = tensor.to(self.device)
            task_vector[name_param] = [ten.to(self.device)-pt_tensor for ten in finetuned_models_dict[name_param]]

        return task_vector

    def process_weight(self,ft_weight:List,tensors:torch.Tensor)->torch.Tensor:
        ft_weights = torch.tensor(ft_weight,dtype=tensors.dtype,device=self.device)
        while len(ft_weights.shape) < len(tensors.shape):
            ft_weights.unsqueeze_(-1)
        return ft_weights

    def save(self, pretrained_model:PreTrainedModel, pretrained_tokenizer:PreTrainedTokenizer):
        dir = self.output_dir
        os.makedirs(dir,exist_ok=True)
        pretrained_model.save_pretrained(dir)
        pretrained_tokenizer.save_pretrained(dir)

    def merge(self):
        
        pass

    def process_list_to_dict(self, param_list)->Dict[str,List[torch.Tensor]]:
        param_dict = {}
        for name_param, tensor in param_list[0].items():
            param_dict[name_param] = [pl[name_param] for pl in param_list]
        return param_dict

# @torch.inference_mode()