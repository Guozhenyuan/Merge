import sys
# sys.path.append('/zju_wck/gzy/MergeLLM/merging_methods/')
# sys.path.append('Merge/merging_methods/')
sys.path.append('.')

from tqdm import tqdm
import torch
from typing import List
from transformers import PreTrainedModel

from task_arithmetic import MergeTaskArithmetic


class MergeTIES(MergeTaskArithmetic):
    def __init__(self, 
                 args, 
                 pretrained_model: PreTrainedModel, 
                 finetuned_models: List[PreTrainedModel]):
        super().__init__(args, pretrained_model, finetuned_models)

        self.ft_weights = args.scaling_coefficient_finetuned
        self.ft_trim_rate = [1-i for i in args.trim_rate_finetuned]

    @torch.no_grad()
    def merge(self):

        # 定义合并后的模型字典
        merged_model_dict = {}

        # 处理微调练模型
        ft_model_dict = self.process_finetuned_models()

        # # 合并模型参数pretrain + lambda * mask( (finetuned model - pretrained_model), TIES_trim )
        for param_name, tensor in tqdm(self.pretrained_model.named_parameters(),total=len(list(self.pretrained_model.named_parameters())),desc="Merging"): # type: ignore
            pt_tensor = tensor.to(self.device)
            ft_tensors = [ftm.to(self.device) for ftm in ft_model_dict[param_name]]
            ft_task_vectors = self.get_task_vectors(pt_tensor,ft_tensors)

            ties_merged_tv = self.process_ties(ft_task_vectors,self.ft_trim_rate)

            merged_tensor = pt_tensor + ties_merged_tv # 合并模型参数pretrain + ties ( task vector )
            merged_model_dict[param_name] = merged_tensor.to('cpu')
            
        return merged_model_dict
    
    # def get_task_vectors(self,pt_tensors:torch.Tensor,ft_tensors:List[torch.Tensor])->torch.Tensor:
    #     task_vectors = torch.stack([tensor - pt_tensors for tensor in ft_tensors],dim=0)
    #     return task_vectors

    def process_ties(self,tv_tensors:torch.Tensor, ks:List[float]) -> torch.Tensor:
        '''
            将输入的任务向量张量经过ties算法处理变成可以直接与微调模型参数合并的张量：
                （1）修剪任务向量，并乘上微调模型的放缩系数
                （2）选择最终的任务向量方向
                （3）使用Disjoint Merge.
        '''
        trimed_tv = torch.stack([self.trim_tensor_by_keep_topk(tv,k) for tv,k in zip(tv_tensors,ks)])
        ft_weight = self.process_weight(self.ft_weights, tv_tensors)
        trimed_tv = trimed_tv * ft_weight # weighted trimed task vector

        mask = self.elect_sign(trimed_tv) 
        merged_tv = self.disjoint_merge(trimed_tv,mask,ft_weight)

        return merged_tv
    

    def disjoint_merge(self, tv_tensors:torch.Tensor, mask:torch.Tensor, ft_weights:torch.Tensor) -> torch.Tensor:
        sumed_tv_tensor = (tv_tensors*mask).sum(dim=0)
        divisor = (ft_weights*mask).sum(dim=0)
        divisor[divisor==0] = 1
        return sumed_tv_tensor/divisor


    def elect_sign(self,tv_tensors:torch.Tensor) -> torch.Tensor:
        majority_sign = tv_tensors.sign().sum(dim=0).sign()
        tv_sign = tv_tensors.sign()
        return tv_sign==majority_sign


    def trim_tensor_by_keep_topk(self,tensor:torch.Tensor,k:float) -> torch.Tensor:
        '''
            保存数值最大的k%个参数，其余参数裁剪为0，参考MergeKit
            tensor：需要裁剪的张量
            return：裁剪后的张量
        '''
        K = int(k * tensor.numel())
        mask = torch.zeros_like(tensor)
        abs_tensor = tensor.abs().view(-1)
        top_K = torch.argsort(abs_tensor,descending=True)[:K]
        mask.view(-1)[top_K]=1

        return mask*tensor