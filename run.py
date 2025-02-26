import sys
# sys.path.append('/zju_wck/gzy/MergeLLM/merging_methods/merge_base')
# sys.path.append('Merge/merging_methods/merge_base')

from utils.loading import embedding_discard, loading_models, add_extra_token
from config.config import parse_args
from merging_methods.linear import MergeLinear

from merging_methods import get_merge_method

args = parse_args()
print(args)
pretrained_model,pretrained_tokenizer,finetuned_models,finetuned_tokenizers = loading_models(args)
new_token_dict = embedding_discard(pretrained_model,pretrained_tokenizer,finetuned_models,finetuned_tokenizers) # type: ignore
print(new_token_dict)

print(type(pretrained_model))

merge_runner = get_merge_method(args=args,
                                pretrained_model=pretrained_model,
                                finetuned_models=finetuned_models)

# merged_model = merge_runner.merge() # type: ignore
merged_model = merge_runner.merge() # type: ignore

# 将额外的token添加到pretrained model中
pt_model, pt_tokenizer = add_extra_token(pretrained_model=pretrained_model,
                                         pretrained_tokenizer=pretrained_tokenizer, # type: ignore
                                         token_dict=new_token_dict,
                                         param_dict=merged_model
                                         )
merge_runner.save(pretrained_model=pt_model,  # type: ignore
                  pretrained_tokenizer=pt_tokenizer) # type: ignore