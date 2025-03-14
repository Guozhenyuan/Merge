import sys
# sys.path.append('/zju_wck/gzy/MergeLLM/merging_methods/')
sys.path.append('.')

from tqdm import tqdm
import torch
from typing import List
from transformers import PreTrainedModel

from merge_base import MergeBase


class MergeTaskArithmetic(MergeBase):
    def __init__(self, 
                 args, 
                 pretrained_model: PreTrainedModel, 
                 finetuned_models: List[PreTrainedModel]):
        super().__init__(args, pretrained_model, finetuned_models)

        self.ft_weights = args.scaling_coefficient_finetuned

    @torch.no_grad()
    def merge(self):

        # 定义合并后的模型字典
        merged_model_dict = {}

        # 处理微调练模型
        ft_model_dict = self.process_finetuned_models()

        # 合并模型参数pretrain + lambda * (finetuned model - pretrained_model)
        for param_name, tensor in tqdm(self.pretrained_model.named_parameters(),total=len(list(self.pretrained_model.named_parameters())),desc="Merging"): # type: ignore
            pt_tensor = tensor.to(self.device)
            ft_tensors = [ftm.to(self.device) for ftm in ft_model_dict[param_name]]
            ft_task_vectors = self.get_task_vectors(pt_tensor,ft_tensors)
            self.get_task_vectors_norm(param_name,pt_tensor,ft_tensors)
            ft_weights = self.process_weight(self.ft_weights,ft_task_vectors)
            # import pdb; pdb.set_trace()
            
            merged_tensor = pt_tensor+(ft_task_vectors*ft_weights).sum(dim=0) 
            merged_model_dict[param_name] = merged_tensor.to('cpu')
        
        return merged_model_dict
    
    def get_task_vectors(self,pt_tensor:torch.Tensor,ft_tensors:List[torch.Tensor])->torch.Tensor:
        task_vectors = torch.stack([tensor - pt_tensor for tensor in ft_tensors],dim=0)
        return task_vectors

    def get_task_vectors_norm(self,param_name:str,pt_tensor:torch.Tensor,ft_tensors:List[torch.Tensor]):
        tast_vectors_norm = [torch.norm(tensor - pt_tensor,p=2) for tensor in ft_tensors]
        print(param_name,tast_vectors_norm)

