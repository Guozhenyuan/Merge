from transformers import AutoModelForCausalLM, AutoTokenizer,PreTrainedModel,PreTrainedTokenizer
from typing import List, Dict, Tuple
import tqdm
import copy
import torch

def loading_models(args):
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_pretrained)
    pretrained_tokenizer = AutoTokenizer.from_pretrained(args.model_pretrained)

    finetuned_models = []
    finetuned_tokenizers = []
    for finetuned_model_path in args.model_finetuned:
        finetuned_models.append(AutoModelForCausalLM.from_pretrained(finetuned_model_path))
        finetuned_tokenizers.append(AutoTokenizer.from_pretrained(finetuned_model_path))

    return pretrained_model,pretrained_tokenizer,finetuned_models,finetuned_tokenizers


def embedding_discard(pretrained_model:PreTrainedModel,
                     pretrained_tokenizer:PreTrainedTokenizer,
                     finetuned_models:List[PreTrainedModel], 
                     finetuned_tokenizers:List[PreTrainedTokenizer]
                     ):
    '''
        如果需要合并的模型中embedding层参数维度不同，进行修剪，直接将微调模型多余的embedding维度剔除
        pretrained_model：表示预训练模型
        finetuned_models：表示微调模型的集合
        return：返裁剪的embedding字典{token:(inputembedding,outputembedding)}表示token对应的embedding向量
    '''

    vocab_num = len(pretrained_tokenizer.get_vocab())
    added_vocab_num = len(pretrained_tokenizer.get_added_vocab())
    
    # 修剪每一个微调模型对应的embedding
    new_token_dict = {}
    for idx, (ft_tokenizer,ft_model) in enumerate(zip(finetuned_tokenizers,finetuned_models)): # type: ignore
        ft_added_vocab_num = len(ft_tokenizer.get_added_vocab())
        if ft_added_vocab_num>added_vocab_num:

            # 获取裁切token对应的embedding
            for token, token_id in list(ft_tokenizer.get_added_vocab().items())[added_vocab_num:ft_added_vocab_num]:
                if token not in new_token_dict.keys():
                    new_token_dict[token] = [(ft_model.get_input_embeddings().weight.data[token_id], ft_model.get_output_embeddings().weight.data[token_id])]
                else:
                    new_token_dict[token] = new_token_dict[token].append((ft_model.get_input_embeddings().weight.data[token_id], ft_model.get_output_embeddings().weight.data[token_id]))

            # 裁切embedding
            embedding_shape1 = ft_model.get_input_embeddings().weight.shape
            ft_model.get_input_embeddings().weight.data = ft_model.get_input_embeddings().weight.data[:vocab_num]
            ft_model.get_output_embeddings().weight.data = ft_model.get_output_embeddings().weight.data[:vocab_num]
            embedding_shape2 = ft_model.get_input_embeddings().weight.shape
            print(ft_model.config._name_or_path,' -before discard:',embedding_shape1, ' -after discard:',embedding_shape2)
    
    return new_token_dict


def add_extra_token(
                    pretrained_model:PreTrainedModel,
                    pretrained_tokenizer:PreTrainedTokenizer,
                    token_dict:Dict[str,List[Tuple[torch.Tensor,torch.Tensor]]],
                    param_dict:Dict[str,torch.Tensor]
                    ):
    '''
        将裁剪的token embedding加载到预训练模型中，并扩展tokenizer维度以及input和output的embedding层维度
        tokenizer：预训练模型对应的tokenizer
        param_dict：合并后模型的参数字典
        token_dict：预处理时裁切掉的token对应的input和output的embedding
    '''
    if token_dict:
        pretrained_model.load_state_dict(param_dict) # 将合并后的模型参数字典加载到预训练模型中
        existing_tokens_num = pretrained_model.get_input_embeddings().weight.shape[0] # 获取预训练模型的现存的token数量
        added_tokens_num = len(token_dict.keys()) # 获取需要添加token字典数量
        pretrained_tokenizer.add_tokens(list(token_dict.keys())) # 向tokenizer中添加额外的字典
        pretrained_model.resize_token_embeddings(existing_tokens_num+added_tokens_num) # 调整预训练模型embedding层维度

        # 提取token_dict中的imput embedding和output embedding参数
        input_embeddings = []
        output_embeddings = []
        for embedding in token_dict.values():
            ie = []
            oe = []
            
            for emb in embedding:
                ie.append(emb[0]) 
                oe.append(emb[1])

            ie_ = sum(ie)/len(ie) 
            oe_ = sum(oe)/len(oe) 

            input_embeddings.append(ie_)
            output_embeddings.append(oe_)   

        input_embeddings = torch.stack(input_embeddings)
        output_embeddings = torch.stack(output_embeddings)   

        # 将修改input embedding和output embedding
        extra_token_id = torch.tensor([pretrained_tokenizer.get_added_vocab()[token] for token in list(token_dict.keys())])
        pretrained_model.get_input_embeddings().weight.data[extra_token_id] = input_embeddings
        pretrained_model.get_output_embeddings().weight.data[extra_token_id] = output_embeddings

        return pretrained_model, pretrained_tokenizer

    else:
        # print(pretrained_model.state_dict().keys())
        # print(param_dict.keys())
        pretrained_model.load_state_dict(param_dict,strict=False)
        return pretrained_model,pretrained_tokenizer

    # # 添加额外单词到预训练模型的字典中
    # for idx, ft_tokenizer in enumerate(finetuned_tokenizers):
    #     pretrained_tokenizer.add_tokens(list(ft_tokenizer.get_added_vocab().keys()))


    # # 将预训练tokenizer添加的单词embedding的tensor值更换为微调模型对应的embedding的值
    # new_added_vocab_num = len(pretrained_tokenizer.get_added_vocab())
    # diff_num = new_added_vocab_num - vocab_num

    # if diff_num != 0: 
    #     for idx, ft_model in enumerate(finetuned_models):
    #         diff_input_embedd = pretrained_model.get_input_embeddings()
            

    # else:
    #     pass

    







def load_model_and_tokenizer_from_hf(path: str):
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model,tokenizer

