import sys


# sys.path.append('/zju_wck/gzy/MergeLLM/merging_methods/')
sys.path.append('merging_methods/')
# sys.path.append('..')

from typing import List,Union
from transformers import PreTrainedModel, PreTrainedTokenizer

from merge_base import MergeBase
from linear import MergeLinear
from task_arithmetic import MergeTaskArithmetic
from ties import MergeTIES
from dare import MergeDARE
from cia import MergeCIA


def get_merge_method(args,
                     pretrained_model: PreTrainedModel, 
                     finetuned_models: List[PreTrainedModel],
                     pretrained_tokenizer: PreTrainedTokenizer,
                     finetuned_tokenizers: List[PreTrainedTokenizer]
                     ): 
    
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
    elif method == 'cia-ta':
        return MergeCIA(args,pretrained_model,finetuned_models,pretrained_tokenizer,finetuned_tokenizers,merged_method='ties')
