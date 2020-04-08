CUDA_VISIBLE_DEVICES="0" python evaluate_task.py \
    --checkpoint=output/CER/hybrid/model_v1_table_0.2_0.6_0.7_30000_1e-4_with_cand_0_seed_1_10000/checkpoint-7500/pytorch_model.bin \
    --result_file=output/CER/hybrid/model_v1_table_0.2_0.6_0.7_30000_1e-4_with_cand_0_seed_1_10000/dev_result.txt \
    --cached_baseline=data/wikisql_entity/dev_result_CER.pkl \
    --task=CER \
    --data_dir=data/wikisql_entity \
    --config_name=configs/table-base-config.json \
