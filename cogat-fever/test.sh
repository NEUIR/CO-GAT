export CUDA_VISIBLE_DEVICES=5


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

#python test.py --checkpoint ./electra/cogat_electra_base/save_model.best.pt \
#--valid_data_path ../data/fever/bert_test.json \
#--model_name_or_path google/electra-base-discriminator \
#--outputpath ./output/cogat_electra_base_test.json \
#--evi_num 6
#python prepare.py \
#--predict ./output/cogat_electra_base_test.json \
#--original ../data/fever/bert_test.json \
#--order ../data/fever/all_test.json \
#--out_file ./blandtest/cogat_electra_base_test.jsonl \


python test.py --checkpoint ./electra/cogat_electra_large/save_model.best.pt \
--valid_data_path ../data/fever/bert_eval.json \
--model_name_or_path google/electra-large-discriminator \
--outputpath ./output/cogat_electra_large.json \
--evi_num 6 \
--hidden_size 1024 \

python results_scorer.py \
--predicted_labels ./output/cogat_electra_large.json \
--predicted_evidence ../data/fever/bert_eval.json \
--actual ../data/fever/dev_eval.json

python test.py --checkpoint ./electra/cogat_electra_large/save_model.best.pt \
--valid_data_path ../data/fever/bert_test.json \
--model_name_or_path google/electra-large-discriminator \
--outputpath ./output/cogat_electra_large_test.json \
--evi_num 6 \
--hidden_size 1024
python prepare.py \
--predict ./output/cogat_electra_large_test.json \
--original ../data/fever/bert_test.json \
--order ../data/fever/all_test.json \
--out_file ./blandtest/cogat_electra_large_test.jsonl \



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

#python test.py --checkpoint ./electra/cogat_without_eviloss_base/save_model.best.pt \
#--valid_data_path ../data/fever/bert_test.json \
#--model_name_or_path google/electra-base-discriminator \
#--outputpath ./output/cogat_without_eviloss_base_test.json \
#--evi_num 6 \
#--ablation
#
#python prepare.py \
#--predict ./output/cogat_without_eviloss_base_test.json \
#--original ../data/fever/bert_test.json \
#--order ../data/fever/all_test.json \
#--out_file ./blandtest/cogat_without_eviloss_base_test.jsonl \


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

#python test.py --checkpoint ./electra/cogat_without_eviloss_large/save_model.best.pt \
#--valid_data_path ../data/fever/bert_test.json \
#--model_name_or_path google/electra-large-discriminator \
#--outputpath ./output/cogat_without_eviloss_large_test.json \
#--evi_num 6 \
#--hidden_size 1024 \
#--ablation
#
#python prepare.py \
#--predict ./output/cogat_without_eviloss_large_test.json \
#--original ../data/fever/bert_test.json \
#--order ../data/fever/all_test.json \
#--out_file ./blandtest/cogat_without_eviloss_large_test.jsonl \
#
#python test.py --checkpoint ./roberta/cogat_roberta_base/save_model.best.pt \
#--valid_data_path ../data/fever/bert_eval.json \
#--model_name_or_path roberta-base \
#--outputpath ./output/cogat_roberta_base.json \
#--evi_num 6 \
#--roberta
#
#python results_scorer.py \
#--predicted_labels ./output/cogat_roberta_base.json \
#--predicted_evidence ../data/fever/bert_eval.json \
#--actual ../data/fever/dev_eval.json


#python test.py --checkpoint ./roberta/cogat_roberta_base/save_model.best.pt \
#--valid_data_path ../data/fever/bert_test.json \
#--model_name_or_path roberta-base \
#--outputpath ./output/cogat_roberta_base_test.json \
#--evi_num 6 \
#--roberta
#python prepare.py \
#--predict ./output/cogat_roberta_base_test.json \
#--original ../data/fever/bert_test.json \
#--order ../data/fever/all_test.json \
#--out_file ./blandtest/cogat_roberta_base_test.jsonl \

#python test.py --checkpoint ./roberta/cogat_roberta_large/save_model.best.pt \
#--valid_data_path ../data/fever/bert_eval.json \
#--model_name_or_path roberta-large \
#--outputpath ./output/cogat_roberta_large.json \
#--evi_num 6 \
#--hidden_size 1024 \
#--roberta
#
#python results_scorer.py \
#--predicted_labels ./output/cogat_roberta_large.json \
#--predicted_evidence ../data/fever/bert_eval.json \
#--actual ../data/fever/dev_eval.json

#python test.py --checkpoint ./roberta/cogat_roberta_large/save_model.best.pt \
#--valid_data_path ../data/fever/bert_test.json \
#--model_name_or_path roberta-large \
#--outputpath ./output/cogat_roberta_large_test.json \
#--evi_num 6 \
#--hidden_size 1024 \
#--roberta
#python prepare.py \
#--predict ./output/cogat_roberta_large_test.json \
#--original ../data/fever/bert_test.json \
#--order ../data/fever/all_test.json \
#--out_file ./blandtest/cogat_roberta_large_test.jsonl \
