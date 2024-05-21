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

#python test.py --checkpoint ./electra/cogat_electra_base/save_model.best.pt \
#--valid_data_path ../data/fever/bert_eval.json \
#--model_name_or_path google/electra-base-discriminator \
#--outputpath ./output/cogat_electra_base.json \
#--evi_num 6 \
##
#python results_scorer.py \
#--predicted_labels ./output/cogat_electra_base.json \
#--predicted_evidence ../data/fever/bert_eval.json \
#--actual ../data/fever/dev_eval.json

#python train.py \
#--train_data_path ../data/fever/bert_train.json \
#--valid_data_path ../data/fever/bert_dev.json \
#--model_name_or_path google/electra-large-discriminator \
#--output_dir ./electra/cogat_electra_large_preattention \
#--num_train_epochs 10 \
#--learning_rate 5e-5 \
#--train_batch_size 8 \
#--valid_batch_size 8 \
#--gradient_accumulation_steps 2 \
#--eval_steps 1000 \
#--patience 5 \
#--evi_num 6 \
#--hidden_size 1024 \
#--pre_attention
#
#python test.py --checkpoint ./electra/cogat_electra_large_preattention/save_model.best.pt \
#--valid_data_path ../data/fever/bert_eval.json \
#--model_name_or_path google/electra-large-discriminator \
#--outputpath ./output/cogat_electra_large_preattention.json \
#--evi_num 6 \
#--hidden_size 1024 \
#
#
#python results_scorer.py \
#--predicted_labels ./output/cogat_electra_large_preattention.json \
#--predicted_evidence ../data/fever/bert_eval.json \
#--actual ../data/fever/dev_eval.json
#
#python train.py \
#--checkpoint ./electra/cogat_electra_large_preattention/save_model.best.pt \
#--train_data_path ../data/fever/bert_train.json \
#--valid_data_path ../data/fever/bert_dev.json \
#--model_name_or_path google/electra-large-discriminator \
#--output_dir ./electra/cogat_electra_large \
#--num_train_epochs 10 \
#--learning_rate 2e-6 \
#--train_batch_size 8 \
#--valid_batch_size 8 \
#--gradient_accumulation_steps 2 \
#--eval_steps 1000 \
#--patience 5 \
#--evi_num 6 \
#--hidden_size 1024 \
#python test.py --checkpoint ./electra/cogat_electra_large/save_model.best.pt \
#--valid_data_path ../data/fever/bert_eval.json \
#--model_name_or_path -google/electra-large-discriminator \
#--outputpath ./output/cogat_electra_large.json \
#--evi_num 6 \
#--hidden_size 1024 \
#
#python results_scorer.py \
#--predicted_labels ./output/cogat_electra_large.json \
#--predicted_evidence ../data/fever/bert_eval.json \
#--actual ../data/fever/dev_eval.json


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
#python test.py --checkpoint ./electra/cogat_without_eviloss_base/save_model.best.pt \
#--valid_data_path ../data/fever/bert_eval.json \
#--model_name_or_path google/electra-base-discriminator \
#--outputpath ./output/cogat_without_eviloss_base.json \
#--evi_num 6 \
#--ablation
#
#python results_scorer.py \
#--predicted_labels ./output/cogat_without_eviloss_base.json \
#--predicted_evidence ../data/fever/bert_eval.json \
#--actual ../data/fever/dev_eval.json

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
#python test.py --checkpoint ./electra/cogat_without_eviloss_preattention_large/save_model.best.pt \
#--valid_data_path ../data/fever/bert_eval.json \
#--model_name_or_path google/electra-large-discriminator \
#--outputpath ./output/cogat_without_eviloss_preattention_large.json \
#--evi_num 6 \
#--hidden_size 1024 \
#--ablation
#
#python results_scorer.py \
#--predicted_labels ./output/cogat_without_eviloss_preattention_large.json \
#--predicted_evidence ../data/fever/bert_eval.json \
#--actual ../data/fever/dev_eval.json
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
#python test.py --checkpoint ./electra/cogat_without_eviloss_large/save_model.best.pt \
#--valid_data_path ../data/fever/bert_eval.json \
#--model_name_or_path google/electra-large-discriminator \
#--outputpath ./output/cogat_without_eviloss_large.json \
#--evi_num 6 \
#--hidden_size 1024 \
#--ablation
#
#python results_scorer.py \
#--predicted_labels ./output/cogat_without_eviloss_large.json \
#--predicted_evidence ../data/fever/bert_eval.json \
#--actual ../data/fever/dev_eval.json

python train.py \
--train_data_path ../data/fever/bert_train.json \
--valid_data_path ../data/fever/bert_dev.json \
--model_name_or_path roberta-base \
--output_dir ./roberta/cogat_roberta_base \
--num_train_epochs 10 \
--learning_rate 5e-5 \
--train_batch_size 16 \
--valid_batch_size 16 \
--eval_steps 1000 \
--patience 5 \
--evi_num 6 \
--roberta

python test.py --checkpoint ./roberta/cogat_roberta_base/save_model.best.pt \
--valid_data_path ../data/fever/bert_eval.json \
--model_name_or_path roberta-base \
--outputpath ./output/cogat_roberta_base.json \
--evi_num 6 \
--roberta

python results_scorer.py \
--predicted_labels ./output/cogat_roberta_base.json \
--predicted_evidence ../data/fever/bert_eval.json \
--actual ../data/fever/dev_eval.json

python train.py \
--train_data_path ../data/fever/bert_train.json \
--valid_data_path ../data/fever/bert_dev.json \
--model_name_or_path roberta-large \
--output_dir ./roberta/cogat_roberta_large_preattention \
--num_train_epochs 10 \
--learning_rate 5e-5 \
--train_batch_size 8 \
--valid_batch_size 8 \
--gradient_accumulation_steps 2 \
--eval_steps 1000 \
--patience 5 \
--evi_num 6 \
--hidden_size 1024 \
--pre_attention \
--roberta

python test.py --checkpoint ./roberta/cogat_roberta_large_preattention/save_model.best.pt \
--valid_data_path ../data/fever/bert_eval.json \
--model_name_or_path roberta-large \
--outputpath ./output/cogat_roberta_large_preattention.json \
--evi_num 6 \
--hidden_size 1024 \
--roberta

python results_scorer.py \
--predicted_labels ./output/cogat_roberta_large_preattention.json \
--predicted_evidence ../data/fever/bert_eval.json \
--actual ../data/fever/dev_eval.json
#
python train.py \
--checkpoint ./roberta/cogat_roberta_large_preattention/save_model.best.pt \
--train_data_path ../data/fever/bert_train.json \
--valid_data_path ../data/fever/bert_dev.json \
--model_name_or_path roberta-large \
--output_dir ./roberta/cogat_roberta_large \
--num_train_epochs 10 \
--learning_rate 2e-6 \
--train_batch_size 8 \
--valid_batch_size 8 \
--gradient_accumulation_steps 2 \
--eval_steps 1000 \
--patience 5 \
--evi_num 6 \
--hidden_size 1024 \
--roberta
#
python test.py --checkpoint ./roberta/cogat_roberta_large/save_model.best.pt \
--valid_data_path ../data/fever/bert_eval.json \
--model_name_or_path roberta-large \
--outputpath ./output/cogat_roberta_large.json \
--evi_num 6 \
--hidden_size 1024 \
--roberta

python results_scorer.py \
--predicted_labels ./output/cogat_roberta_large.json \
--predicted_evidence ../data/fever/bert_eval.json \
--actual ../data/fever/dev_eval.json