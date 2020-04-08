OUTPUT_DIR=hybrid/model_v1_table_0.2_0.4_0.7_30000_1e-4_with_cand_0
CUDA_VISIBLE_DEVICES="1" python run_table_HR_finetuning.py \
    --output_dir=output/HR/$OUTPUT_DIR"test" \
    --model_name_or_path=output/$OUTPUT_DIR \
    --model_type=HR \
    --do_train \
    --data_dir=data/wikisql_entity \
    --evaluate_during_training \
    --per_gpu_train_batch_size=64 \
    --per_gpu_eval_batch_size=64 \
    --gradient_accumulation_steps=1 \
    --learning_rate=3e-5 \
    --num_train_epochs=80 \
    --save_total_limit=10 \
    --seed=1 \
    --seed_num=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --config_name=configs/table-base-config.json \
    --save_steps=2500 \
    --logging_steps=500 #> /dev/null 2>&1 &

# OUTPUT_DIR=data/pre-trained_models/tiny-bert/2nd_General_TinyBERT_4L_312D
# CUDA_VISIBLE_DEVICES="0" python run_table_HR_finetuning.py \
#     --output_dir=output/HR/bert_seed_1 \
#     --model_name_or_path=$OUTPUT_DIR \
#     --model_type=HR \
#     --do_train \
#     --data_dir=data/wikisql_entity \
#     --evaluate_during_training \
#     --per_gpu_train_batch_size=64 \
#     --per_gpu_eval_batch_size=64 \
#     --gradient_accumulation_steps=1 \
#     --learning_rate=3e-5 \
#     --num_train_epochs=80 \
#     --save_total_limit=10 \
#     --seed=1 \
#     --seed_num=1 \
#     --cache_dir=cache \
#     --overwrite_output_dir \
#     --config_name=configs/table-base-config.json \
#     --save_steps=2500 \
#     --logging_steps=500 \
#     --is_bert > /dev/null 2>&1 &