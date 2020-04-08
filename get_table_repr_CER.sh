OUTPUT_DIR=model_v1_table_0.2_0.2_30000_1e-4_with_dist_cand_2
CUDA_VISIBLE_DEVICES="1" python run_table_CER_finetuning.py \
    --output_dir=output/CER/$OUTPUT_DIR \
    --model_name_or_path=output/CER/model_v1_table_0.2_0.2_30000_1e-4_with_dist_cand_2/checkpoint-90000/pytorch_model.bin \
    --model_type=CER \
    --data_dir=data/wikisql_entity \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --seed=1 \
    --cache_dir=cache \
    --max_entity_candidate=10000 \
    --config_name=configs/table-base-config.json \
    --use_cand \
    --sample_distribution \
    --get_table_repr