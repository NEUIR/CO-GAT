export CUDA_VISIBLE_DEVICES=4

python test.py --checkpoint ./electra_large/save_model.best.pt \
--model_name_or_path ../../plm/googleelectra_large_discriminator \
--outputpath ./output/electra_large.json \
--corpus ../data/scifact/corpus.jsonl \
--evidence_retrieval ../data/scifact/prediction/rationale_selection_dev_scibert_mlm.jsonl \
--dataset ../data/scifact/claims_dev.jsonl \
--evi_num 6 \
--hidden_size 1024 \

python3 merge_predictions.py \
    --rationale-file ../data/scifact/prediction/rationale_selection_dev_scibert_mlm.jsonl \
    --label-file ./output/electra_large.json \
    --result-file ./output/merged_predictions_electra_large.jsonl
#
##
python ./metric.py \
    --gold ../data/scifact/claims_dev.jsonl \
    --corpus ../data/scifact/corpus.jsonl \
    --prediction ./output/merged_predictions_electra_large.jsonl

#
python test.py --checkpoint ./electra_base/save_model.best.pt \
--model_name_or_path ../../plm/googleelectra_base_discriminator \
--outputpath ./output/electra_base.json \
--corpus ../data/scifact/corpus.jsonl \
--evidence_retrieval ../data/scifact/prediction/rationale_selection_dev_scibert_mlm.jsonl \
--dataset ../data/scifact/claims_dev.jsonl \
--evi_num 6 \

python3 merge_predictions.py \
    --rationale-file ../data/scifact/prediction/rationale_selection_dev_scibert_mlm.jsonl \
    --label-file ./output/electra_base.json \
    --result-file ./output/merged_predictions_electra_base.jsonl
#
#
python ./metric.py \
    --gold ../data/scifact/claims_dev.jsonl \
    --corpus ../data/scifact/corpus.jsonl \
    --prediction ./output/merged_predictions_electra_base.jsonl
#
python test.py --checkpoint ./electra_base/save_model.best.pt \
--model_name_or_path ../../plm/googleelectra_base_discriminator \
--outputpath ./output/electra_base_test.json \
--corpus ../data/scifact/corpus.jsonl \
--evidence_retrieval ../data/scifact/prediction/rationale_selection_test.jsonl \
--dataset ../data/scifact/claims_test.jsonl \
--evi_num 6 \

python3 merge_predictions.py \
    --rationale-file ../data/scifact/prediction/rationale_selection_test.jsonl \
    --label-file ./output/electra_base_test.json \
    --result-file ./output/merged_predictions_electra_base_test.jsonl



python test.py --checkpoint ./electra_large/save_model.best.pt \
--model_name_or_path ../../plm/googleelectra_large_discriminator \
--outputpath ./output/electra_large_test.json \
--corpus ../data/scifact/corpus.jsonl \
--evidence_retrieval ../data/scifact/prediction/rationale_selection_test.jsonl \
--dataset ../data/scifact/claims_test.jsonl \
--evi_num 6 \
--hidden_size 1024

python3 merge_predictions.py \
    --rationale-file ../data/scifact/prediction/rationale_selection_test.jsonl \
    --label-file ./output/electra_large_test.json \
    --result-file ./output/merged_predictions_electra_large_test.jsonl



