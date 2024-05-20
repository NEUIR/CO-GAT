export CUDA_VISIBLE_DEVICES=4

python train.py \
--train_data_path ./data/train_cogat.json \
--valid_data_path ./data/dev_cogat.json \
--model_name_or_path google/electra-base-discriminator \
--output_dir ./electra_base \
--num_train_epochs 10 \
--learning_rate 5e-5 \
--train_batch_size 16 \
--valid_batch_size 16 \
--eval_steps 1000 \
--patience 5 \
--evi_num 6 \
--hidden_size 768 \

python train.py \
--train_data_path ./data/train_cogat.json \
--valid_data_path ./data/dev_cogat.json \
--model_name_or_path google/electra-large-discriminator \
--output_dir ./electra_large_preattention \
--num_train_epochs 10 \
--learning_rate 5e-5 \
--train_batch_size 16 \
--valid_batch_size 16 \
--gradient_accumulation_steps 1 \
--eval_steps 1000 \
--patience 5 \
--evi_num 6 \
--hidden_size 1024 \
--pre_attention

#
python train.py \
--checkpoint ./electra_large_preattention/save_model.best.pt \
--train_data_path ./data/train_cogat.json \
--valid_data_path ./data/dev_cogat.json \
--model_name_or_path google/electra-large-discriminator \
--output_dir ./electra_large \
--num_train_epochs 10 \
--learning_rate 2e-6 \
--train_batch_size 8 \
--valid_batch_size 8 \
--gradient_accumulation_steps 2 \
--eval_steps 1000 \
--patience 5 \
--evi_num 6 \
--hidden_size 1024 \



