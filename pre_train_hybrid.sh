OUTPUT_DIR=output/hybrid/no_random/model_v1_table_0.2_0.6_0.7_5000_5e-5_with_cand_0
CUDA_VISIBLE_DEVICES="0" python run_hybrid_table_lm_finetuning.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=hybrid \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --data_dir=data/wikisql_entity \
    --evaluate_during_training \
    --mlm \
    --mlm_probability=0.2 \
    --ent_mlm_probability=0.6 \
    --mall_probability=0.7 \
    --per_gpu_train_batch_size=20 \
    --per_gpu_eval_batch_size=20 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-5 \
    --num_train_epochs=40 \
    --save_total_limit=10 \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --max_entity_candidate=5000 \
    --config_name=configs/table-base-config.json \
    --save_steps=5000 \
    --logging_steps=500 \
    --use_cand \
    --exclusive_ent=0 \
    --resume=output/hybrid/model_v1_table_0.2_0.6_0.7_30000_1e-4_with_cand_0/pytorch_model.bin
    # --random_sample
    #> /dev/null 2>&1 &