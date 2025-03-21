# PAD: Personalized Alignment at Decoding-time


This repository contains the official implementation of the paper PAD: Personalized Alignment at Decoding-time (https://openreview.net/pdf?id=e7AUJpP8bV).
This paper presents Personalized Alignment at Decoding-time (PAD), a novel framework designed to align LLM outputs with diverse personalized preferences during the inference phase, eliminating the need for additional training. 

Acknowledgement: This repository is built based on https://github.com/deeplearning-wisc/args.


## Setup
The following packages, and versions were used:

```bash=
git clone https://github.com/deeplearning-wisc/args.git

conda create -n PAD python=3.10 -y
conda activate PAD

cd args
pip -r requirements.txt
```
We recommend the following versions for the main packages:

| Mandatory    | Recommend |
| ------------ |-----------|
| python       | 3.10      |
| torch        | 2.1.0     |
| transformers | 4.45.0    |
| datasets     | 2.16.0    |
| accelerate   | 0.34.2    |
| peft         | 0.14.0    |
| trl          | 0.9.6     |

## Training

The training module of PAD is coming soon.


## Inference
We have released our Personalized Reward Model at https://huggingface.co/RuizheChen/PAD, which can be directly employed to perform alignment at decoding-time.

The following command can be run to start personalized generation:

```bash
cd inference

python collect_model_outs.py \
    --run_percent 100. \
    --config="configs/psoups_llama3SFT_hhh.config" \
    --out_file="results/psoups_llama3SFT_hhh" \
    --llm_gpu="cuda:0" \
    --rm_gpu="cuda:0" \
    --llm="princeton-nlp/Llama-3-Base-8B-SFT" \
    --rm="RuizheChen/PAD" \
    --dataset="psoups" \
    --sys_prompt="[Guidelines] Your task is to generate response by considering the following principle. \
[Principles] harmless and helpfulness and humor \
[Instruction] "
```

The result of the generation will be stored as a jsonl file with the path `results/psoups_llama3SFT_hhh_0.jsonl`. We have provided config files for aligning with different dimensions on both datasets in `./configs`. For a new alignment task, we recommend to perform grid search with `inference/configs/grid_search.config`.

Detailed codes for reproducing experiments are provided in `inference/decoding.sh`.

## Evaluations

To prepare for the evaluations, please extract your models output in the following form and save it as `results/your_run_name.jsonl`. Note that the response should contain both prompt and the output generated by the model.

```jsonld
[
    {
        "prompt": "Are you okay? You look",
        "response": "Are you okay? You look a bit tired or stressed. Anything you'd like to talk about?",
        "method": "greedy"
    },
    {
        "prompt": "...",
        "response": "...",
        "method": "..."
    },
    ...
]
```

To run the evaluations, execute the following commands:

```bash


# helpfulness

python measure_reward.py \
    --out_file="<your_results>.jsonl" \
    --tokenizer="Ray2333/gpt2-large-helpful-reward_model" \
    --rm="Ray2333/gpt2-large-helpful-reward_model" \
    --rm_gpu="cuda:0"

python measure_reward.py \
    --out_file="<your_results>.jsonl" \
    --tokenizer="RLHFlow/ArmoRM-Llama3-8B-v0.1" \
    --rm="RLHFlow/ArmoRM-Llama3-8B-v0.1" \
    --rm_gpu="cuda:0" \
    --dimension=0

# harmless

python measure_reward.py \
    --out_file="<your_results>.jsonl" \
    --tokenizer="Ray2333/gpt2-large-harmless-reward_model" \
    --rm="Ray2333/gpt2-large-harmless-reward_model" \
    --rm_gpu="cuda:0"

python measure_reward.py \
    --out_file="<your_results>.jsonl" \
    --tokenizer="RLHFlow/ArmoRM-Llama3-8B-v0.1" \
    --rm="RLHFlow/ArmoRM-Llama3-8B-v0.1" \
    --rm_gpu="cuda:0" \
    --dimension=10

# humor

python measure_reward.py \
  --out_file="<your_results>.jsonl" \
  --tokenizer="mohameddhiab/humor-no-humor" \
  --rm="mohameddhiab/humor-no-humor" \
  --rm_gpu="cuda:0"
```


## Citation

If you find this repository useful in your research, please consider citing:

```
@inproceedings{chen2024pad,
  title={Pad: Personalized alignment at decoding-time},
  author={Chen, Ruizhe and Zhang, Xiaotian and Luo, Meng and Chai, Wenhao and Liu, Zuozhu},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2024}
}
```
