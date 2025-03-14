# python Merge/try.py --output=$C --device=$device --merge_method=$mm --model_pretrained=$pt \
#                     --model_finetuned $A $B \
#                     --scaling_coefficient_finetuned 0.5 0.5  \
#                     --trim_rate_finetuned 0.7 0.7 \
#                     --drop_rate_finetuned 0.7 0.7 \




M="results/merged_llm/ICM"
device="cuda:1"
mm="cia-ta"
pt="Qwen/Qwen2.5-7B"
A="Qwen/Qwen2.5-Math-7B-Instruct"
B="Qwen/Qwen2.5-Coder-7B-Instruct"
C="Qwen/Qwen2.5-7B-Instruct"
grad_path="results/grad/ICM"
chat="datasets/math.json"
math="datasets/code.json"
code="datasets/chat.json"
merge_it=3
pt_drop=0.5

python run.py --output=$M --device=$device --merge_method=$mm --model_pretrained=$pt\
                    --model_finetuned $A $B $C\
                    --scaling_coefficient_finetuned 0.1\
                    --trim_rate_finetuned 0.7 0.7\
                    --drop_rate_finetuned 0.5 0.5 0.5\
                    --output_grad $grad_path\
                    --calibration_datasets_path $math $code $chat\
                    --merge_iteration $merge_it\
                    --drop_rate_pretrained $pt_drop\
                    --sparse_method topk