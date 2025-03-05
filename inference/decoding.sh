#!/bin/sh


python collect_model_outs.py \
    --run_percent 100. \
    --config="configs/vanilla_decoding.config" \
    --out_file="baseline_results/psoups_llama3SFT" \
    --llm_gpu="cuda:0" \
    --rm_gpu="cuda:0" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --sys_prompt=""

python collect_model_outs.py \
    --run_percent 100. \
    --config="configs/psoups_llama3SFT_helpfulness.config" \
    --out_file="results_1dim/psoups_llama3SFT_helpfulness" \
    --llm_gpu="cuda:0" \
    --rm_gpu="cuda:0" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. \
[Principles] helpfulness \
[Instruction] " &

python collect_model_outs.py \
    --run_percent 100. \
    --config="configs/psoups_llama3SFT_harmless.config" \
    --out_file="results_1dim/psoups_llama3SFT_harmless" \
    --llm_gpu="cuda:1" \
    --rm_gpu="cuda:1" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. \
[Principles] harmless \
[Instruction] " &

python collect_model_outs.py \
    --run_percent 100. \
    --config="configs/psoups_llama3SFT_humor.config" \
    --out_file="results_1dim/psoups_llama3SFT_humor" \
    --PAD="PAD4" \
    --llm_gpu="cuda:2" \
    --rm_gpu="cuda:2" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. \
[Principles] humor \
[Instruction] "


python collect_model_outs.py \
    --run_percent 100. \
    --config="configs/psoups_llama3SFT_harmlesshumor.config" \
    --out_file="results_2dim/psoups_llama3SFT_harmlesshumor" \
    --llm_gpu="cuda:0" \
    --rm_gpu="cuda:0" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. \
[Principles] harmless and humor \
[Instruction] "

python collect_model_outs.py \
    --run_percent 100. \
    --config="configs/psoups_llama3SFT_helpfulnesshumor.config" \
    --out_file="results_2dim/psoups_llama3SFT_helpfulnesshumor" \
    --llm_gpu="cuda:1" \
    --rm_gpu="cuda:1" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. \
[Principles] helpfulness and humor \
[Instruction] "

python collect_model_outs.py \
    --run_percent 100. \
    --config="configs/psoups_llama3SFT_harmlesshelpfulness.config" \
    --out_file="results_2dim/psoups_llama3SFT_harmlesshelpfulness" \
    --llm_gpu="cuda:2" \
    --rm_gpu="cuda:2" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. \
[Principles] harmless and helpfulness \
[Instruction] "

python collect_model_outs.py \
    --run_percent 100. \
    --config="configs/psoups_llama3SFT_hhh.config" \
    --out_file="results_3dim/psoups_llama3SFT_hhh" \
    --llm_gpu="cuda:2" \
    --rm_gpu="cuda:2" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. \
[Principles] harmless and helpfulness and humor \
[Instruction] "

echo "All commands executed"