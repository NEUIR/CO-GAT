export CUDA_VISIBLE_DEVICES=5
export TRANSFORMERS_CACHE=/home/users/lyq/cache

#python test.py --checkpoint ./electra/cogat_electra_base/save_model.best.pt \
#--valid_data_path ../kgat_data/bert_test.json \
#--model_name_or_path google/electra-base-discriminator \
#--outputpath ./output/cogat_electra_base_test.json \
#--evi_num 6
#python prepare.py \
#--predict ./output/cogat_electra_base_test.json \
#--original ../kgat_data/bert_test.json \
#--order ../kgat_data/all_test.json \
#--out_file ./blandtest/cogat_electra_base_test.jsonl \
#
#python test.py --checkpoint ./electra/cogat_electra_large/save_model.best.pt \
#--valid_data_path ../kgat_data/bert_test.json \
#--model_name_or_path google/electra-large-discriminator \
#--outputpath ./output/cogat_electra_large_test.json \
#--evi_num 6 \
#--hidden_size 1024
#python prepare.py \
#--predict ./output/cogat_electra_large_test.json \
#--original ../kgat_data/bert_test.json \
#--order ../kgat_data/all_test.json \
#--out_file ./blandtest/cogat_electra_large_test.jsonl \


python test.py --checkpoint ./electra/cogat_without_eviloss_base/save_model.best.pt \
--valid_data_path ../kgat_data/bert_test.json \
--model_name_or_path google/electra-base-discriminator \
--outputpath ./output/cogat_without_eviloss_base_test.json \
--evi_num 6 \
--ablation

python prepare.py \
--predict ./output/cogat_without_eviloss_base_test.json \
--original ../kgat_data/bert_test.json \
--order ../kgat_data/all_test.json \
--out_file ./blandtest/cogat_without_eviloss_base_test.jsonl \

python test.py --checkpoint ./electra/cogat_without_eviloss_large/save_model.best.pt \
--valid_data_path ../kgat_data/bert_test.json \
--model_name_or_path google/electra-large-discriminator \
--outputpath ./output/cogat_without_eviloss_large_test.json \
--evi_num 6 \
--hidden_size 1024 \
--ablation

python prepare.py \
--predict ./output/cogat_without_eviloss_large_test.json \
--original ../kgat_data/bert_test.json \
--order ../kgat_data/all_test.json \
--out_file ./blandtest/cogat_without_eviloss_large_test.jsonl \


python test.py --checkpoint ./roberta/cogat_roberta_base/save_model.best.pt \
--valid_data_path ../kgat_data/bert_test.json \
--model_name_or_path roberta-base \
--outputpath ./output/cogat_roberta_base_test.json \
--evi_num 6 \
--roberta
python prepare.py \
--predict ./output/cogat_roberta_base_test.json \
--original ../kgat_data/bert_test.json \
--order ../kgat_data/all_test.json \
--out_file ./blandtest/cogat_roberta_base_test.jsonl \

python test.py --checkpoint ./roberta/cogat_roberta_large/save_model.best.pt \
--valid_data_path ../kgat_data/bert_test.json \
--model_name_or_path roberta-large \
--outputpath ./output/cogat_roberta_large_test.json \
--evi_num 6 \
--hidden_size 1024 \
--roberta
python prepare.py \
--predict ./output/cogat_roberta_large_test.json \
--original ../kgat_data/bert_test.json \
--order ../kgat_data/all_test.json \
--out_file ./blandtest/cogat_roberta_large_test.jsonl \
