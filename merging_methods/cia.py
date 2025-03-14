import sys

# # sys.path.append('/zju_wck/gzy/MergeLLM/merging_methods/')
# # sys.path.append('Merge/merging_methods/')
sys.path.append('.')

import gc
from tqdm import tqdm
import torch
import os
from typing import List, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from ties import MergeTIES
from compute_grad import compute_grad
from utils.utils import load_json

class MergeCIA(MergeTIES):
    def __init__(self, 
                 args, 
                 pretrained_model: PreTrainedModel, 
                 finetuned_models: List[PreTrainedModel],
                 pretrained_tokenizer: PreTrainedTokenizer,
                 finetuned_tokenizers: List[PreTrainedTokenizer],
                #  calibration_datasets: List[str],
                 merged_method: str = 'ta'):
        super().__init__(args, pretrained_model, finetuned_models)

        self.ft_drop_rate = args.drop_rate_finetuned
        self.pt_drop_rate = args.drop_rate_pretrained
        self.sparse_method = args.sparse_method
        self.cd_paths = args.calibration_datasets_path
        self.merge_it = args.merge_iteration

        self.args = args
        self.merged_method = merged_method
        self.pretrained_tokenizer = pretrained_tokenizer
        self.finetuned_tokenizers = finetuned_tokenizers
        

    
    def merge(self):

        # 定义合并后的模型字典
        merged_model_dict = {}

        # 处理微调练模型
        ft_model_dict = self.process_finetuned_models()

        # 加载校准数据集
        cal_datasets = self.get_calibration_datasets()
        
        # 加载不同的chat_templates
        chat_templates = self.get_chat_templates()

        # 循环迭代
        cur_model_name_or_path = self.args.model_pretrained # 加载地址
        for i in tqdm(range(self.merge_it)):
            cur_model_dict = self.merge_once(cal_datasets, chat_templates, ft_model_dict, i, cur_model_name_or_path) # type: ignore
            self.pretrained_model.load_state_dict(cur_model_dict)

            # 保存当前模型
            save_dir = self.output_dir + '/iter_{}'.format(i)
            self.pretrained_model.save_pretrained(save_dir)
            self.pretrained_tokenizer.save_pretrained(save_dir)

            # 加载下一次迭代的模型
            cur_model_name_or_path = save_dir

            # 保存合并的模型字典
            merged_model_dict = cur_model_dict
        
        return merged_model_dict

    
    def get_chat_templates(self):
        ct_list = []
        for ct in self.finetuned_tokenizers:
            ct_list.append(ct.chat_template)
        return ct_list
        
    def get_calibration_datasets(self):
        cd = []
        for path in self.cd_paths:
            data = load_json(path)
            cd.append(data)
        return cd
    

    def merge_once(self, cal_datasets, chat_templates, ft_model_dict, iter_num, cur_model_name_or_path):

        model = AutoModelForCausalLM.from_pretrained(cur_model_name_or_path, trust_remote_code=True, torch_dtype="bfloat16")
        tokenizer = AutoTokenizer.from_pretrained(cur_model_name_or_path, trust_remote_code=True)

        # 计算pretrained model在不同校准数据集上的梯度
        cd_grads = []
        for i,(cd,ct) in enumerate(zip(cal_datasets, chat_templates)):

            # 梯度保存路径
            dir_grad = self.args.output_grad + '/' + 'ft_' + str(i) + '/' + 'iter_' + str(iter_num)
            if not os.path.exists(dir_grad):
                os.makedirs(dir_grad)

            # 计算梯度
            grad = compute_grad(cd, self.args.device_parallel_size, model, tokenizer, self.args.calibration_batch_size,
                                self.args.max_seq_length, ct, dir_grad, save=True)
            cd_grads.append(grad)
        
        cd_grad_dict = self.process_list_to_dict(cd_grads) # {param_name: List[cd1,cd2,cd3]}
        
        # # 计算finetuned model上的wandb值
        # pass

        with torch.no_grad():
            next_model_param = {}
            for param_name, tensor in tqdm(self.pretrained_model.named_parameters(),total=len(list(self.pretrained_model.named_parameters())),desc="Merging"): # type: ignore
                pt_tensor = tensor.to(self.device)
                ft_tensors = [ftm.to(self.device) for ftm in ft_model_dict[param_name]]
                grad_tensors = [cdg.to(self.device) for cdg in cd_grad_dict[param_name]]

                ft_task_vectors = self.get_task_vectors_sparse(pt_tensor,ft_tensors,grad_tensors,self.pt_drop_rate,self.sparse_method)

                if self.merged_method == 'ta':
                    ft_weights = self.process_weight(self.ft_weights,ft_task_vectors)
                    merged_tensor = pt_tensor+(ft_task_vectors*ft_weights).sum(dim=0) 
                    # del ft_weights
                    # gc.collect()
                    # torch.cuda.empty_cache()
                elif self.merged_method == 'ties':
                    ties_merged_tv = self.process_ties(ft_task_vectors,self.ft_trim_rate)
                    merged_tensor = pt_tensor + ties_merged_tv # 合并模型参数pretrain + ties ( task vector )
                    # del ties_merged_tv
                    # gc.collect()
                    # torch.cuda.empty_cache()

                next_model_param[param_name] = merged_tensor.to('cpu')
                # tensor.to('cpu')
                # [ftm.to('cpu') for ftm in ft_model_dict[param_name]]
                # [cdg.to('cpu') for cdg in cd_grad_dict[param_name]]
                # del pt_tensor,ft_tensors,grad_tensors,ft_task_vectors,merged_tensor
                # gc.collect()
                torch.cuda.empty_cache()
                # print(torch.cuda.memory_allocated()/1024**2)
                # print(torch.cuda.memory_reserved()/1024**2)

                # # 获取所有 GPU Tensor
                # gpu_tensors = [obj for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda]

                # print(f"GPU 上的 Tensor 数量: {len(gpu_tensors)}")
                # for i, tensor in enumerate(gpu_tensors):
                #     print(f"Tensor {i}: 形状 {tensor.shape}, 设备 {tensor.device}, 占用 {tensor.element_size() * tensor.numel() / 1024**2:.2f} MB")

                # # import pdb;pdb.set_trace()
        return next_model_param

    def get_task_vectors_sparse(self, 
                                pt_tensor: torch.Tensor, 
                                ft_tensors: List[torch.Tensor], 
                                grad_tensors: List[torch.Tensor], 
                                # wandb_tensors: List[torch.Tensor], 
                                drop: float, 
                                method:str) -> torch.Tensor:
        
        '''
            这里用来计算稀疏化后的结果
        '''

        task_vectors = []
        ptt=pt_tensor
        dp=drop

        ptt=pt_tensor
        for ftt,gt in zip(ft_tensors,grad_tensors):
            tv = ftt - ptt
            if method == 'random':
                task_vectors.append((1-torch.bernoulli(torch.full_like(input=ptt, fill_value=dp)))*(tv)) # 使用bernoulli分布进行随机mask，并进行放缩
            elif method == 'topk':
                original_shape = tv.shape
                gt = gt.flatten()
                num_mask_param = int(dp*gt.numel())
                kth_values, _ = gt.abs().kthvalue(k=num_mask_param, dim=0, keepdim=True)
                task_vectors.append(((gt.abs()>=kth_values).reshape(original_shape))*tv)
            elif method == 'bottom':
                original_shape = tv.shape
                gt = gt.flatten()
                num_mask_param = int((1-dp)*gt.numel())
                kth_values, _ = gt.abs().kthvalue(k=num_mask_param, dim=0, keepdim=True)
                task_vectors.append(((gt.abs()<=kth_values).reshape(original_shape))*tv)
        
        result = torch.stack(task_vectors,dim=0)
        # del gt,task_vectors
        # gc.collect()
        # torch.cuda.empty_cache()
        return result
