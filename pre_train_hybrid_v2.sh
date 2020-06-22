OUTPUT_DIR=output/hybrid/v2/model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam_no_visibility_test
CUDA_VISIBLE_DEVICES="1" python run_hybrid_table_lm_finetuning.py \
    --output_dir=$OUTPUT_DIR \
    --model_type=hybrid \
    --model_name_or_path=bert-base-uncased \
    --do_train \
    --data_dir=data/wikitables_v2 \
    --evaluate_during_training \
    --mlm \
    --mlm_probability=0.2 \
    --ent_mlm_probability=0.6 \
    --mall_probability=0.7 \
    --per_gpu_train_batch_size=25 \
    --per_gpu_eval_batch_size=25 \
    --gradient_accumulation_steps=2 \
    --learning_rate=1e-4 \
    --num_train_epochs=100 \
    --save_total_limit=10 \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --max_entity_candidate=10000 \
    --config_name=configs/table-base-config_v2.json \
    --save_steps=5000 \
    --logging_steps=1000 \
    --use_cand \
    --exclusive_ent=0 \
    --no_visibility \
    --random_sample #> /dev/null 2>&1 &
    # --resume=output/hybrid/v2/model_v1_table_0.2_0.4_0.7_10000_1e-4_cand_0_adam/checkpoint-35000/pytorch_model.bin \