python Merge/try.py --output=$C --device=$device --merge_method=$mm --model_pretrained=$pt \
                    --model_finetuned $A $B \
                    --scaling_coefficient_finetuned 0.5 0.5  \
                    --trim_rate_finetuned 0.7 0.7 \
                    --drop_rate_finetuned 0.7 0.7 \