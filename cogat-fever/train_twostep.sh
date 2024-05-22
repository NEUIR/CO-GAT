export CUDA_VISIBLE_DEVICES=0

#python train.py \
#--train_data_path ../data/fever/bert_train.json \
#--valid_data_path ../data/fever/bert_dev.json \
#--model_name_or_path google/electra-base-discriminator \
#--output_dir ./electra/cogat_electra_base \
#--num_train_epochs 10 \
#--learning_rate 5e-5 \
#--train_batch_size 16 \
#--valid_batch_size 16 \
#--eval_steps 1000 \
#--patience 5 \
#--evi_num 6 \


python train.py \
--train_data_path ../data/fever/bert_train.json \
--valid_data_path ../data/fever/bert_dev.json \
--model_name_or_path google/electra-large-discriminator \
--output_dir ./electra/cogat_electra_large_preattention \
--num_train_epochs 10 \
--learning_rate 5e-5 \
--train_batch_size 8 \
--valid_batch_size 8 \
--gradient_accumulation_steps 2 \
--eval_steps 1000 \
--patience 5 \
--evi_num 6 \
--hidden_size 1024 \
--pre_attention

python train.py \
--checkpoint ./electra/cogat_electra_large_preattention/save_model.best.pt \
--train_data_path ../data/fever/bert_train.json \
--valid_data_path ../data/fever/bert_dev.json \
--model_name_or_path google/electra-large-discriminator \
--output_dir ./electra/cogat_electra_large \
--num_train_epochs 10 \
--learning_rate 2e-6 \
--train_batch_size 8 \
--valid_batch_size 8 \
--gradient_accumulation_steps 2 \
--eval_steps 1000 \
--patience 5 \
--evi_num 6 \
--hidden_size 1024 \



#python train.py \
#--train_data_path ../data/fever/bert_train.json \
#--valid_data_path ../data/fever/bert_dev.json \
#--model_name_or_path google/electra-base-discriminator \
#--output_dir ./electra/cogat_without_eviloss_base \
#--num_train_epochs 10 \
#--learning_rate 5e-5 \
#--train_batch_size 16 \
#--valid_batch_size 16 \
#--eval_steps 1000 \
#--patience 5 \
#--evi_num 6 \
#--ablation
#


#python train.py \
#--train_data_path ../data/fever/bert_train.json \
#--valid_data_path ../data/fever/bert_dev.json \
#--model_name_or_path google/electra-large-discriminator \
#--output_dir ./electra/cogat_without_eviloss_preattention_large \
#--num_train_epochs 10 \
#--learning_rate 5e-5 \
#--train_batch_size 8 \
#--valid_batch_size 8 \
#--gradient_accumulation_steps 2 \
#--eval_steps 1000 \
#--patience 5 \
#--evi_num 6 \
#--hidden_size 1024 \
#--pre_attention \
#--ablation

#
#python train.py \
#--checkpoint ./electra/cogat_without_eviloss_preattention_large/save_model.best.pt \
#--train_data_path ../data/fever/bert_train.json \
#--valid_data_path ../data/fever/bert_dev.json \
#--model_name_or_path google/electra-large-discriminator \
#--output_dir ./electra/cogat_without_eviloss_large \
#--num_train_epochs 10 \
#--learning_rate 2e-6 \
#--train_batch_size 8 \
#--valid_batch_size 8 \
#--gradient_accumulation_steps 2 \
#--eval_steps 1000 \
#--patience 5 \
#--evi_num 6 \
#--hidden_size 1024 \
#--ablation


#python train.py \
#--train_data_path ../data/fever/bert_train.json \
#--valid_data_path ../data/fever/bert_dev.json \
#--model_name_or_path roberta-base \
#--output_dir ./roberta/cogat_roberta_base \
#--num_train_epochs 10 \
#--learning_rate 5e-5 \
#--train_batch_size 16 \
#--valid_batch_size 16 \
#--eval_steps 1000 \
#--patience 5 \
#--evi_num 6 \
#--roberta

#python train.py \
#--train_data_path ../data/fever/bert_train.json \
#--valid_data_path ../data/fever/bert_dev.json \
#--model_name_or_path roberta-large \
#--output_dir ./roberta/cogat_roberta_large_preattention \
#--num_train_epochs 10 \
#--learning_rate 5e-5 \
#--train_batch_size 8 \
#--valid_batch_size 8 \
#--gradient_accumulation_steps 2 \
#--eval_steps 1000 \
#--patience 5 \
#--evi_num 6 \
#--hidden_size 1024 \
#--pre_attention \
#--roberta


#python train.py \
#--checkpoint ./roberta/cogat_roberta_large_preattention/save_model.best.pt \
#--train_data_path ../data/fever/bert_train.json \
#--valid_data_path ../data/fever/bert_dev.json \
#--model_name_or_path roberta-large \
#--output_dir ./roberta/cogat_roberta_large \
#--num_train_epochs 10 \
#--learning_rate 2e-6 \
#--train_batch_size 8 \
#--valid_batch_size 8 \
#--gradient_accumulation_steps 2 \
#--eval_steps 1000 \
#--patience 5 \
#--evi_num 6 \
#--hidden_size 1024 \
#--roberta
##
