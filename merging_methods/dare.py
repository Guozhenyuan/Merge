import sys

from linear import MergeLinear
# sys.path.append('/zju_wck/gzy/MergeLLM/merging_methods/')
# sys.path.append('Merge/merging_methods/')
sys.path.append('.')

from tqdm import tqdm
import torch
from typing import List
from transformers import PreTrainedModel


from ties import MergeTIES


class MergeDARE(MergeTIES):
    def __init__(self, 
                 args, 
                 pretrained_model: PreTrainedModel, 
                 finetuned_models: List[PreTrainedModel],
                 merged_method: str = 'ta'):
        super().__init__(args, pretrained_model, finetuned_models)

        self.ft_drop_rate = args.drop_rate_finetuned
        self.sparse_method = args.sparse_method
        self.merged_method = merged_method

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
            ft_task_vectors = self.get_task_vectors_sparse(pt_tensor,ft_tensors,self.ft_drop_rate, self.sparse_method)

            if self.merged_method == 'ta':
                ft_weights = self.process_weight(self.ft_weights,ft_task_vectors)
                merged_tensor = pt_tensor+(ft_task_vectors*ft_weights).sum(dim=0) 
            elif self.merged_method == 'ties':
                ties_merged_tv = self.process_ties(ft_task_vectors,self.ft_trim_rate)
                merged_tensor = pt_tensor + ties_merged_tv # 合并模型参数pretrain + ties ( task vector )

            merged_model_dict[param_name] = merged_tensor.to('cpu')
            
        return merged_model_dict
    
    def get_task_vectors_sparse(self, 
                                pt_tensor: torch.Tensor, 
                                ft_tensors: List[torch.Tensor], 
                                drop_ps:List[float], 
                                method:str) -> torch.Tensor:
        '''
            将任务向量进行稀疏化，返回稀疏化后的任务向量
            pt_tensor：表示预训练模型中某层参数
            ft_tensors：表示参与合并的所有模型中某层参数
            drop_p：表示丢弃参数的比率
            method：表示稀疏化的方式
            return：返回一个tensor，表示所有参与合并的模型参数
        '''

        task_vectors = []
        # for ptt,ftt,dp in zip(pt_tensor,ft_tensors,drop_ps):
        ptt=pt_tensor
        for ftt,dp in zip(ft_tensors,drop_ps):
            if method == 'random':
                # import pdb;pdb.set_trace()
                task_vectors.append(((1-torch.bernoulli(torch.full_like(input=ptt, fill_value=dp)))*(ftt-ptt))/(1-dp)) # 使用bernoulli分布进行随机mask，并进行放缩
                # task_vectors.append((1-torch.bernoulli(torch.full_like(input=ptt, fill_value=dp)))*(ftt-ptt))

            elif method == 'magnitude': # 将最小的一部分参数进行随机mask
                tv = ftt - ptt
                original_shape = tv.shape
                tv = tv.flatten()
                num_mask_param = int(dp*tv.numel())
                kth_values, _ = tv.abs().kthvalue(k = num_mask_param, dim=0, keepdim=True)
                task_vectors.append(((~(tv.abs()<=kth_values))*tv).reshape(original_shape)/(1-dp))
        
        return torch.stack(task_vectors,dim=0)
                

                



        