import sys
sys.path.append('.')

import re
import json
from typing import List
from utils.utils import load_json, save_json


def eval_math(inference_datas:List):

    '''
        input: a list of inference datas who need to be eval
            {
                ct: []
                ans: str
                pred: str
            }
        output: acc, format acc, and a dict inference_datas add match_pred
    '''
    
    # 正则表达式
    pattern = re.compile(r'\\boxed\{(.*?)\}')

    # 计算正确个数，以及匹配到该格式的数量
    total = len(inference_datas)
    correct = 0
    correct_f = 0

    new_inf_datas = []

    for inf_d in inference_datas:
        ans = inf_d['ans']
        pred = inf_d['pred']

        match = pattern.search(pred)
        if match:
            correct_f = correct_f + 1
            match_pred = match.group(1)

            if match_pred == ans:
                correct = correct + 1

            new_inf_datas.append({'ct':inf_d['ct'], 'ans':ans, 'pred':pred, 'match_pred':match_pred})
        else:
            new_inf_datas.append({'ct':inf_d['ct'], 'ans':ans, 'pred':pred, 'match_pred':None})

    acc = correct/total
    acc_f = correct_f/total

    return acc, acc_f, new_inf_datas


if __name__ == "__main__":
    file_path = "/zju_wck/czc/dataset/gsm8k_grad_dare.json"
    inference_data = load_json(file_path)
    acc, acc_f, results = eval_math(inference_data)
    print("acc:{} acc:{}".format(acc,acc_f))
    save_json(results,'./results/try.json')
