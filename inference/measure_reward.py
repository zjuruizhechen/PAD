from transformers import AutoTokenizer, AutoModelForSequenceClassification, LlamaTokenizer, LlamaForSequenceClassification
import argparse
import torch
import json
import re
from LLaMAFactory.src.llamafactory.model.loader import load_model
from LLaMAFactory.src.llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments
from transformers import Seq2SeqTrainingArguments



parser = argparse.ArgumentParser()
parser.add_argument("--out_file", type=str)
parser.add_argument("--rm", type=str)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--rm_gpu", type=str, default="cuda:0")
parser.add_argument("--npout", type=str, default="")
parser.add_argument("--experiment", type=str, default="hhrlhf")
parser.add_argument("--start_index", type=int, help='Start index for file pattern', default=0)
parser.add_argument("--end_index", type=int, help='End index for file pattern', default=0)
parser.add_argument("--dimension", type=int, help='End index for file pattern', default=None)

args = parser.parse_args()
data_args = DataArguments
training_args = Seq2SeqTrainingArguments
finetuning_args = FinetuningArguments
model_args = ModelArguments
model_args.model_name_or_path = args.rm

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

num_labels = 2 if 'humor' in args.rm else 1

if "ArmoRM" in args.rm:
    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm, trust_remote_code=True, torch_dtype=torch.bfloat16).to(args.rm_gpu)
elif "PAD" in args.rm:
    rm_model = load_model(tokenizer, model_args, finetuning_args, training_args.do_eval, add_valuehead=True).to(args.rm_gpu)
else:
    rm_model = AutoModelForSequenceClassification.from_pretrained(args.rm, num_labels=num_labels, torch_dtype=torch.float16).to(args.rm_gpu)

def extract_out(output_data):
    # output = output_data["result"]
    # if output.startswith(": "): output = output[2:]
    # output = re.split("human:", output, flags=re.IGNORECASE)[0]
    # return output_data["prompt"] + output
    if "result" in output_data:
        output = output_data["result"]
    elif "output" in output_data:
        output = output_data["output"]
    elif "response" in output_data:
        output = output_data["response"]

    if args.experiment == "hhrlhf":
        output_np = output.removeprefix(output_data["prompt"])
        if output_np.startswith(": "): output = output_np[2:]
        output_np = re.split("human:", output_np, flags=re.IGNORECASE)[0]
        # return output_data["prompt"]+output_np
        if "system\n\nuser\n\n" in output:
            new_output = output.replace("system\n\nuser\n\n", "")
        else:
            new_output = output
        return new_output
    elif args.experiment == "shp":
        return output

    # return output_data["output"]

def get_rm(text, dimension=None):
    if "ArmoRM" in args.rm or "PAD" in args.rm:
        if "ArmoRM" in args.rm:
            assist_idx = text.find("assistant\n\n")
            r_accept = text[assist_idx + 11:].strip()

            # human_idx = text.rfind("\n\nHuman:")
            query = text[: assist_idx].strip()
            text_idx2 = text.find("[Instruction]")
            query = query.replace(query[:text_idx2 + 13], "")

            messages = [{"role": "user", "content": query},
                        {"role": "assistant", "content": r_accept}]
        else:
            assist_idx = text.find("assistant\n\n")
            r_accept = text[assist_idx + 11:].strip()
            # human_idx = text.rfind("\n\nHuman:")
            query = text[: assist_idx].strip()

            messages = [{"role": "user", "content": query},
                        {"role": "assistant", "content": r_accept}]

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(args.rm_gpu)

        output = rm_model(input_ids)
        if "ArmoRM" in args.rm:
            # rm_out = output.score
            rm_out = output.rewards[0][dimension]
        else:
            rm_out = output[2][0][-1]
        rm_val = rm_out.flatten().item()
        del rm_out
        del input_ids
        return rm_val

    else:
        text = text.replace("assistant", " Assistant:", 1)
        text = text.replace("user", "Human:", 1)
        text_idx = text.find("[Guidelines]")
        text_idx2 = text.find("[Instruction]")
        text = text.replace(text[text_idx:text_idx2+13], "")
        # print(text)
        # exit()
        tokens = tokenizer(text, return_tensors="pt").input_ids.to(args.rm_gpu)
        # print(f"{tokens.shape=}", tokens.shape)
        # 1966 1819 1813
        # print(tokens.shape)
        if tokens.shape[1] >= 1024: return None
        if 'humor' in args.rm:
            if tokens.shape[1] >= 512:
                tokens = tokens[:, :512]
        rm_out = rm_model(tokens)
        # print(rm_out)
        if 'humor' in args.rm:
            rm_val = rm_out.logits.flatten()[1].item()
        else:
            rm_val = rm_out.logits.flatten().item()
        # print(rm_val)
        del rm_out
        del tokens
        return rm_val

def get_rm_from_tokens(tokens):
    return rm_model(torch.tensor(tokens).unsqueeze(0).to(args.rm_gpu)).logits.flatten().item()
import numpy as np
from tqdm import tqdm

if "{}" not in args.out_file:
    with open(args.out_file, "r") as out_f:
        lines = json.load(out_f)
    rm_scores = []
    num_skip = 0
    for line in tqdm(lines):
        outp = extract_out(line)
        # print(outp)
        if len(outp) == 0: rm_scores.append(0.)
        # print(f"{get_rm(outp)}")
        rm_score = get_rm(outp, args.dimension)
        if rm_score == None:
            print("skipped one")
            num_skip += 1
            continue
        else: rm_scores.append(rm_score)


    if args.npout != "": np.save(f"{args.npout}", np.array(rm_scores))
    print(f"{np.mean(rm_scores)=}")
    print(f"{num_skip=}")

else:
    for idx in range(args.start_index, args.end_index + 1):
        out_file = args.out_file.format(idx)
        print(f"Processing {out_file}")

        with open(out_file, "r") as out_f:
            lines = json.load(out_f)

        rm_scores = []
        num_skip = 0
        for line in tqdm(lines):
            outp = extract_out(line)
            if len(outp) == 0:
                rm_scores.append(0.)
                continue

            rm_score = get_rm(outp, args.dimension)
            if rm_score is None:
                print("Skipped one")
                num_skip += 1
                continue
            else:
                rm_scores.append(rm_score)

        print(f"Mean RM Score for {out_file}: {np.mean(rm_scores)}")
        print(f"Number of skipped entries: {num_skip}")
