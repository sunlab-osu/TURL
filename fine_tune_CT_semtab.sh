OUTPUT_DIR=model_v1_table_0.2_0.6_0.7_10000_1e-4_candnew_0_adam
MODE=4
CUDA_VISIBLE_DEVICES="1" python run_table_CT_finetuning.py \
    --output_dir=output/CT/Semtab/wiki_train70/$MODE/$OUTPUT_DIR \
    --model_name_or_path=output/hybrid/v2/$OUTPUT_DIR \
    --model_type=CT \
    --do_train \
    --data_dir=data/Semtab \
    --evaluate_during_training \
    --per_gpu_train_batch_size=32 \
    --per_gpu_eval_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --learning_rate=5e-6 \
    --cls_learning_rate=5e-5 \
    --num_train_epochs=70 \
    --save_total_limit=10 \
    --seed=1 \
    --cache_dir=cache \
    --overwrite_output_dir \
    --config_name=configs/table-base-config_v2.json \
    --save_steps=100 \
    --logging_steps=100 \
    --warmup_steps=1000 \
    --mode=$MODE #> /dev/null 2>&1 &
