OUTPUT_DIR=hybrid/model_v1_table_0.2_0.4_0.7_30000_1e-4_with_cand_0
CUDA_VISIBLE_DEVICES="0" python run_table_TR_finetuning.py \
    --output_dir=output/TR/$OUTPUT_DIR \
    --model_name_or_path=output/$OUTPUT_DIR \
    --model_type=TR \
    --do_train \
    --data_dir=data/WebQueryTable_Dataset \
    --evaluate_during_training \
    --per_gpu_train_batch_size=8 \
    --per_gpu_eval_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --learning_rate=5e-5 \
    --num_train_epochs=80 \
    --save_total_limit=10 \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --config_name=configs/table-base-config.json \
    --save_steps=1000 \
    --logging_steps=1000 \
    --neg_num=5