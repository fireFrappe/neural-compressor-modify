#!/bin/bash
set -x

function main {

  init_params "$@"
  run_tuning

}
# init params
function init_params {
  tuned_checkpoint=saved_results
  tokenizer_name=bert-large-uncased-whole-word-masking-finetuned-squad
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --output_model=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      --debug) debug="--debug";;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done
}


# run_tuning
function run_tuning {
    if [[ "${topology}" == "bert_large_ipex" ]]; then
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
        python run_qa.py \
            --model_name_or_path $model_name_or_path \
            --dataset_name squad \
            --do_eval \
            --max_seq_length 384 \
            --no_cuda \
            --tune \
            --output_dir $tuned_checkpoint \
            ${extra_cmd}
    fi
    if [[ "${topology}" == "distilbert_base_ipex" ]]; then
        model_name_or_path="distilbert-base-uncased-distilled-squad"
        python run_qa.py \
           --model_name_or_path $model_name_or_path \
           --dataset_name squad \
           --do_eval \
           --max_seq_length 384 \
           --no_cuda \
           --tune \
           --output_dir $tuned_checkpoint \
           ${extra_cmd}
    fi
    if [[ "${topology}" == "bert_large_1_10_ipex" ]]; then
        pip install transformers==3.0.2
        python run_qa_1_10.py \
            --model_type bert \
            --model_name_or_path $input_model \
            --do_lower_case \
            --predict_file $dataset_location \
            --tokenizer_name $tokenizer_name \
            --do_eval \
            --max_seq_length 384 \
            --doc_stride 128 \
            --no_cuda \
            --tune \
            --output_dir $tuned_checkpoint \
            --int8 \
            --int8_fp32
    fi 
}

main "$@"
