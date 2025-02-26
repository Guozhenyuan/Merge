import sys


# sys.path.append('/zju_wck/gzy/MergeLLM/merging_methods/')
sys.path.append('Merge/merging_methods/')

from typing import List,Union
from transformers import PreTrainedModel

from merge_base import MergeBase
from linear import MergeLinear
from task_arithmetic import MergeTaskArithmetic
from ties import MergeTIES
from dare import MergeDARE


def get_merge_method(args,
                     pretrained_model: PreTrainedModel, 
                     finetuned_models: List[PreTrainedModel]): 
    
    method = args.merge_method
    if method == 'linear':
        return MergeLinear(args,pretrained_model,finetuned_models)
    elif method == 'task-arithmetic':
        return MergeTaskArithmetic(args,pretrained_model,finetuned_models)
    elif method == 'ties':
        return MergeTIES(args,pretrained_model,finetuned_models)
    elif method == 'dare-task-arithmetic':
        return MergeDARE(args,pretrained_model,finetuned_models,merged_method='ta')
    elif method == 'dare-ties':
        return MergeDARE(args,pretrained_model,finetuned_models,merged_method='ties')
