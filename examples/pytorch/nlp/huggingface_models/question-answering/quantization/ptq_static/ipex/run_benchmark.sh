#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

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
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark"
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    extra_cmd=""
    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
    fi
    echo $extra_cmd
    if [[ "${topology}" == "bert_large_ipex" ]]; then
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
        python run_qa.py \
            --model_name_or_path $model_name_or_path \
            --dataset_name squad \
            --do_eval \
            --max_seq_length 384 \
            --no_cuda \
            --output_dir $tuned_checkpoint \
            $mode_cmd \
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
            --output_dir $tuned_checkpoint \
            $mode_cmd \
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
            --output_dir $tuned_checkpoint \
            $mode_cmd \
            ${extra_cmd}
    fi
}


main "$@"
