from typing import List
import torch
from torch.nn import functional as F
from tqdm import tqdm
from LLaMAFactory.src.llamafactory.model.loader import load_model
from LLaMAFactory.src.llamafactory.hparams import DataArguments, FinetuningArguments, ModelArguments
from transformers import Seq2SeqTrainingArguments

data_args = DataArguments
training_args = Seq2SeqTrainingArguments
finetuning_args = FinetuningArguments

# import the huggingface transformers libraries
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaForSequenceClassification
# from trl import AutoModelForCausalLMWithValueHead
#### auto size stuff
import numpy as np
def factors(x):
    return [i for i in range(1,x+1) if x%i==0]

def auto_size(seq_len, topk):
    estimated = (28672/(seq_len*1.5)) -11.52605
    # hack
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)): return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]
###

def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))

# From huggingface
def rcache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past

def even_chunk(data, chunk_size=10):
    assert data.shape[0] % chunk_size == 0, "chunk_size must evenly divide the topk"
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:(i+chunk_size)]

# reward based search
class ARGS:
    def __init__(self, llm_path, rm_path, llm_dev="cuda:0", rm_dev="cuda:1", torch_dtype=torch.float16):
        self.llm_dev = llm_dev
        self.llm_path = llm_path
        self.rm_path = rm_path
        self.rm_dev = rm_dev
        print("Loading LLM...")
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype).to(self.llm_dev)
        self.LLM.eval()
        self.model_args = ModelArguments
        self.model_args.model_name_or_path = rm_path

        
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.LLM.pad_token = self.tokenizer.eos_token
        if "Llama-3" not in llm_path:
            self.tokenizer2 = AutoTokenizer.from_pretrained(rm_path)
            self.tokenizer2.pad_token = self.tokenizer2.eos_token
        
        print("Loading RM...")
        self.RM = load_model(self.tokenizer, self.model_args, finetuning_args, training_args.do_eval,
                                 add_dpovaluehead=True).to(self.rm_dev)
        self.RM.eval()
        
    def get_input_ids(self, prompt: str) -> torch.Tensor:
        messages = [
            {"role": "user", "content": prompt},
        ]
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_tensors="pt").to(self.llm_dev)
        return tokens
    
    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def generate_greedy_step_large(self, mout, input_ids, pre_screen_beam_width=40, weight=0., rm_cached=None, chunk_size=10, debug=True, _use_cache=True):
        out_logits = mout.logits[:, -1]

        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        if debug: print(f"{expanded_tis.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        if debug: print(f"{to_rm_eval.shape=}")
        if debug: print(f"{out_logits.shape[0] * pre_screen_beam_width=}")
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        if debug: print(f"{flat_trme.shape=}")
        
        new_rm_cached = None
        current_best_score = None
        current_best_tokens = None
        if debug: print(f"{prescreen_logits.flatten().shape=}")

        for chunk, chunk_logits in zip(even_chunk(flat_trme.to(self.rm_dev), chunk_size), even_chunk(prescreen_logits.flatten(), chunk_size)):
            pkv = None if not _use_cache else rm_cached

            rm_out = self.RM(**self.LLM.prepare_inputs_for_generation(input_ids=chunk, attention_mask=create_attention_mask(chunk.shape[1], chunk.shape[0]).to(self.rm_dev), past_key_values=pkv, use_cache=True))
            current_rm_cached = rm_out.past_key_values
            rewards = rm_out.logits.flatten().to(self.llm_dev)
            del rm_out
            if debug: print(f"{rewards=}")
            if debug: print(f"{rewards.shape=}")

            new_scores = rewards * weight + chunk_logits
            if debug: print(f"{new_scores=}")
            
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
            current_score = new_scores[top_k_ids[0]].item()
            if debug: print(f"{current_score=} {current_best_score=} ")
            if (current_best_score is None) or (current_score > current_best_score):
                if debug: print(f"Updated!!")
                
                current_best_score = current_score
                current_best_tokens = chunk.to(self.llm_dev)[top_k_ids]
                new_rm_cached = self.LLM._reorder_cache(current_rm_cached, top_k_ids.repeat(chunk_size,))
            
        if debug: print(f"{new_scores.shape=}")
        
        return current_best_tokens, new_rm_cached
        
    def generate_step(self, mout, input_ids, pre_screen_beam_width=40, weight=0., method="greedy", temperature=0.7, rm_cached=None, debug=True):
        out_logits = mout.logits[:, -1]
        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        if weight==0:
            expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1).to(self.llm_dev)
            if debug: print(f"{expanded_tis.shape=}")

            to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
            if debug: print(f"{to_rm_eval.shape=}")

            if debug: print(f"{out_logits.shape[0] * pre_screen_beam_width=}")
            flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
            if debug: print(f"{flat_trme.shape=}")

            new_scores = prescreen_logits.flatten()
            if method == "greedy":
                _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
            elif method == "topk":
                # assume B=1
                assert input_ids.shape[0] == 1
                new_scores = new_scores / temperature
                scores = F.softmax(new_scores, dim=-1)
                top_k_ids = torch.multinomial(scores, num_samples=1)
            else:
                raise ValueError(f"Invalid method '{method}'")

            if debug: print(f"{top_k_ids.shape=}")
            # rm_cached = self.LLM._reorder_cache(rm_cached, top_k_ids.repeat(pre_screen_beam_width,))
            if debug: print(f"{rewards[top_k_ids]=}")

            return flat_trme[top_k_ids], rm_cached

        if "Llama-3" not in self.llm_path:
            text_llm = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            input_ids_rm = self.tokenizer2.encode_plus(text_llm, return_tensors="pt", add_special_tokens=True)['input_ids'].to(self.rm_dev)

            text_out_logits = [self.tokenizer.decode([token], skip_special_tokens=False) for token in prescreen_tokens[0]]

            text_out_logits = [' ' if x == '' else x for x in text_out_logits]

            prescreen_tokens_rm = [self.tokenizer2.encode(word, add_special_tokens=False)[0] for word in text_out_logits]

            prescreen_tokens_rm = torch.tensor([prescreen_tokens_rm], device=self.llm_dev)  # Using [data] to create a 2D tensor


        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1).to(self.llm_dev)
        if debug: print(f"{expanded_tis.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        if debug: print(f"{to_rm_eval.shape=}")

        if debug: print(f"{out_logits.shape[0] * pre_screen_beam_width=}")
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        if debug: print(f"{flat_trme.shape=}")


        if "Llama-3" in self.llm_path:
            input_ids = input_ids.to(self.rm_dev)
            rm_out = self.RM(**self.LLM.prepare_inputs_for_generation(input_ids=input_ids,
                                                                     attention_mask=create_attention_mask(input_ids.shape[1],
                                                                                                          input_ids.shape[
                                                                                                              0]).to(
                                                                         self.rm_dev), past_key_values=None,
                                                                     use_cache=True))
            rm_logits = rm_out[0][:, -1]

            prescreen_rm_logits = rm_logits[0, prescreen_tokens[0].to(self.rm_dev)]

            dpo_logps = prescreen_rm_logits.log_softmax(-1)
            ref_logps = prescreen_logits.log_softmax(-1)

            logp_dpo_ref = (dpo_logps.to(self.llm_dev) - ref_logps.to(self.llm_dev)).squeeze(0)

            rewards = logp_dpo_ref.to(self.llm_dev)
        else:
            input_ids_rm = input_ids_rm.to(self.rm_dev)
            rm_out = self.RM(**self.LLM.prepare_inputs_for_generation(input_ids=input_ids_rm,
                                                                     attention_mask=create_attention_mask(input_ids_rm.shape[1],
                                                                                                          input_ids_rm.shape[
                                                                                                              0]).to(
                                                                         self.rm_dev), past_key_values=None,
                                                                     use_cache=True))
            # print(rm_out)
            rm_logits = rm_out[0][:, -1]
            prescreen_rm_logits = rm_logits[0, prescreen_tokens_rm[0].to(self.rm_dev)]
            rewards = prescreen_rm_logits.to(self.llm_dev) - prescreen_logits.flatten().to(self.llm_dev)


        del rm_out
        if debug: print(f"{rewards.shape=}")
        new_scores = rewards * weight + prescreen_logits.flatten()
        
        if debug: print(f"{new_scores.shape=}")

        if method == "greedy":
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
        elif method == "topk":
            # assume B=1
            assert input_ids.shape[0] == 1
            new_scores = new_scores / temperature
            scores = F.softmax(new_scores, dim=-1)
            top_k_ids = torch.multinomial(scores, num_samples=1)
        else:
            raise ValueError(f"Invalid method '{method}'")
            
        if debug: print(f"{top_k_ids.shape=}")
        if debug: print(f"{rewards[top_k_ids]=}")

        return flat_trme[top_k_ids], rm_cached
    
    def generate(self, prompt, weight=0., topk=1, max_new_token=128, method="greedy", temperature=0.7, chunk_size=5, debug=False):
        tokens = self.get_input_ids(prompt)
        initial_len = tokens.shape[-1]
        if chunk_size == "auto":
            chunk_size = auto_size(initial_len + max_new_token, topk)
            print(f"auto {chunk_size=}, {topk=}, {initial_len=}!")
        
        if tokens.shape[-1] > self.LLM.config.to_dict().get("max_sequence_length", 2048):
            print("The sequence of tokens is too long!!! Returning none!")
            return None
        
        if tokens.shape[-1] > self.RM.config.to_dict().get("max_sequence_length", 2048):
            print("The sequence of tokens is too long!!! Returning none!")
            return None
          
        rm_cached = None
        cached = None
        
        iterator_obj = range(max_new_token)
        if debug: iterator_obj = tqdm(iterator_obj)
        for _ in iterator_obj:
            if debug: print(f"{type(cached)=}")
            if debug: print(f"{type(rm_cached)=}")
            with torch.no_grad():
                if cached is None:
                    mout = self.LLM(**self.LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), past_key_values=None, use_cache=True))

                else:
                    mout = self.LLM(**self.LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), past_key_values=cached, use_cache=True))

                
                if method == "greedy_large":
                    if debug: print("large")
                    tokens, rm_cached = self.generate_greedy_step_large(mout, tokens, topk, weight, rm_cached, chunk_size, debug)   
                else:
                    tokens, rm_cached = self.generate_step(mout, tokens, topk, weight, method, temperature, rm_cached, debug)
                del mout

        return tokens
