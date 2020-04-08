OUTPUT_DIR=model_v1_table_0.2_0.4_0.7_30000_1e-4_with_cand_0
MODE=4
CUDA_VISIBLE_DEVICES="2" python run_table_CT_finetuning.py \
    --output_dir=output/CT/remove_dev/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/hybrid/$OUTPUT_DIR \
    --model_type=CT \
    --do_train \
    --data_dir=data/wikisql_entity \
    --evaluate_during_training \
    --per_gpu_train_batch_size=20 \
    --per_gpu_eval_batch_size=20 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --num_train_epochs=25 \
    --save_total_limit=10 \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --config_name=configs/table-base-config.json \
    --save_steps=1000 \
    --logging_steps=500 \
    --mode=$MODE > /dev/null 2>&1 &
