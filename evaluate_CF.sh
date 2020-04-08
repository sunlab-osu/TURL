CUDA_VISIBLE_DEVICES="0" python evaluate_task.py \
    --checkpoint=output/hybrid/model_v1_table_0.2_0.6_0.7_30000_1e-4_with_cand_0 \
    --task=CF \
    --data_dir=data/wikisql_entity \
    --config_name=configs/table-base-config.json \
