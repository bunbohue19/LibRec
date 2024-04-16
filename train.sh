export CUDA_VISIBLE_DEVICES=0
python run_classification.py \
    --model_name_or_path  facebook/bart-base \
    --train_file '/users/anhld/LibRec/train.csv' \
    --validation_file '/users/anhld/LibRec/val.csv' \
    --test_file '/users/anhld/LibRec/test.csv' \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "text" \
    --label_column_name "labels" \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 1024 \
    --per_device_train_batch_size 2 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/ \
    --overwrite_output_dir \