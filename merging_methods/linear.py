import sys
# sys.path.append('/zju_wck/gzy/MergeLLM/merging_methods/')
sys.path.append('Merge/merging_methods/')

from tqdm import tqdm
import torch
from typing import List
from transformers import PreTrainedModel

from merge_base import MergeBase


class MergeLinear(MergeBase):
    def __init__(self, 
                 args, 
                 pretrained_model: PreTrainedModel, 
                 finetuned_models: List[PreTrainedModel]):
        super().__init__(args, pretrained_model, finetuned_models)

        self.ft_weight = args.scaling_coefficient_finetuned

    @torch.no_grad()
    def merge(self):

        # 定义合并后的模型字典
        merged_model_dict = {}

        # 处理微调练模型
        ft_model_dict = self.process_finetuned_models()

        # 使用线性加权平均方式进行合并
        for param_name, tensor in tqdm(self.pretrained_model.named_parameters(),total=len(list(self.pretrained_model.named_parameters())),desc="Merging"): # type: ignore
            # pt_tensor = tensor.to(self.device)
            ft_tensors = torch.stack([ftm.to(self.device) for ftm in ft_model_dict[param_name]],dim=0)
            ft_weights = self.process_weight(self.ft_weight,ft_tensors)
            # import pdb; pdb.set_trace()
            merged_tensor = (ft_tensors*ft_weights).sum(dim=0)
            merged_model_dict[param_name] = merged_tensor.to('cpu')
        
        return merged_model_dict
            