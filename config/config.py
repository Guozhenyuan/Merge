import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Interface for merging LLMs")

    # 基础参数
    parser.add_argument('--seed', type=int, default=777, help='Repeat experiments with random seeds')
    parser.add_argument('--device', type=str, default='cuda:1', help='The GPU selected during the merge')

    # 合并相关参数
    parser.add_argument('--merge_method', type=str, default='average', help='The methods for model merging')
    # parser.add_argument('--scaling_coefficient', type=float, default=0.4, help='The scaling coefficient for each fine-tuned model is set as pretrained + scaling_coefficient * delta, with the default value set to 0.4.')
    # parser.add_argument('--trim_rate', type=float, default=0.8, help=f'In the TIES method, the trim rate used is defaulted to top-20%. This trims 80% of the delta parameters based on their magnitude.')
    # parser.add_argument('--drop_rate', type=str, default=0.9, help='In the DARE method, the default ratio for dropping delta parameters is 90%.')
    parser.add_argument('--sparse_method', type=str, default='random', help='Method for sparsifying task vectors in DARE/CIA')

    # 模型相关参数
    parser.add_argument('--model_pretrained', type=str, default='meta-llama/Llama-2-7b-hf', help='Pre-trained base model')
    parser.add_argument('--model_finetuned', type=str, nargs='+', help='The list of fine-tuned models')
    parser.add_argument('--trim_rate_finetuned', type=float, nargs='+', help=f'Assign a trim ratio to each fine-tuned delta')
    parser.add_argument('--drop_rate_finetuned', type=float, nargs='+', help='Assign a drop ratio to each fine-tuned delta.')
    parser.add_argument('--scaling_coefficient_finetuned', type=float, nargs='+', help='Assign scaling coefficient to each fine-tuned model/delta.')
    
    # cia相关计算参数
    parser.add_argument('--output_grad', type=str, default='results/grad/', help='Dircetory for save grad')
    parser.add_argument('--calibration_datasets_path', type=str, nargs='+', help='The path list of calibration dataset')
    parser.add_argument('--calibration_batch_size', type=int, default=4, help='calibration data batch size for compute grad')
    parser.add_argument('--max_seq_length',type=int, default=1024, help='input paddinged length')
    parser.add_argument('--merge_iteration', type=int, default=2, help='The counts of merged llm iteration')
    parser.add_argument('--drop_rate_pretrained', type=float, default=0.5, help='Assign a drop ratio to each pretrained llm.')
    parser.add_argument('--device_parallel_size', type=int, default=2, help='The gpu number used')

    # 输出结果相关参数
    parser.add_argument('--output', type=str, default='./output', help='Directory for merged model outputs')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)