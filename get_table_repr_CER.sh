OUTPUT_DIR=model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam
CUDA_VISIBLE_DEVICES="1" python run_table_CER_finetuning.py \
    --output_dir=output/CER/hybrid/v2/$OUTPUT_DIR \
    --model_name_or_path=output/hybrid/v2/$OUTPUT_DIR \
    --model_type=CER \
    --data_dir=data/wikitables_v2 \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --seed=1 \
    --cache_dir=cache \
    --max_entity_candidate=10000 \
    --config_name=configs/table-base-config_v2.json \
    --get_table_repr