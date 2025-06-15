import argparse
import json
import math
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from multiprocessing import Process, Queue
import sys

SIMPLE_CHAT_TEMPLATE = "{% for message in messages %}{{message['role'].capitalize() + ': ' + message['content'] + '\n\n'}}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

def flatten_nested_list(dataset: list):
    final_data = []
    for data in dataset:
        for d in data:
            final_data.append(d)
    print(len(final_data))
    return final_data

def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def process_segment(segment, reward_model, eval_mode, gpu_id, result_queue, target_key):
    # Set the visible GPU for this process.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Now load the tokenizer and model; note that because of CUDA_VISIBLE_DEVICES,
    # the single visible GPU will be "cuda:0" inside this process.
    tokenizer = AutoTokenizer.from_pretrained(reward_model)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Load the model and move it to GPU.
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model,
        cache_dir="/home/ubuntu/jonghoon/~/.cache/huggingface/hub",
    ).to("cuda")

    if eval_mode == "initial":
        batch_size = 1
        idx = 0
        # Create a tqdm progress bar for this segment.
        with tqdm(total=len(segment), desc=f"GPU {gpu_id} Processing", leave=True) as pbar:
            while idx < len(segment):
                start = idx
                end = min(idx + batch_size, len(segment))
                convs = [
                    [
                        {"role": "user", "content": data['prompt']},
                        {"role": "assistant", "content": data[target_key].strip()},
                    ]
                    for data in segment[start:end]
                ]
                templated_convs = tokenizer.apply_chat_template(
                    convs, tokenize=False, add_generation_prompt=False, add_specifal_tokens=False,
                )
                templated_convs = tokenizer(templated_convs, return_tensors="pt").to(model.device)
                context_length = templated_convs['input_ids'].shape[1]
                lm_backbone = getattr(model, model.base_model_prefix)
                attention_mask = templated_convs['input_ids'] != tokenizer.pad_token_id
                position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
                input_ids = torch.masked_fill(templated_convs['input_ids'], ~attention_mask, 0)
                with torch.no_grad():
                    responses = lm_backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_dict=True,
                        output_hidden_states=True,
                        use_cache=False,  # otherwise mistral-based RM would error out
                    )
                reward_logits = model.score(responses.hidden_states[-1])
                logits = reward_logits[
                    torch.arange(reward_logits.size(0), device=reward_logits.device),
                    context_length - 1,
                ].squeeze(-1)
                for i in range(end - start):
                    segment[start + i][f'{target_key}_reward'] = logits[i].item()
                idx += batch_size
                pbar.update(end - start)

    # After processing, put the segment into the result queue.
    result_queue.put(segment)

def main(args):
    # Load and flatten the dataset (unchanged)
    with open(args.save_path, "r", encoding='utf-8') as f:
        dataset = json.load(f)
    if isinstance(dataset[0], list):
        dataset = flatten_nested_list(dataset)

    dataset = dataset[:200]
    # Split dataset into segments (unchanged)
    gpu_ids = [0, 1]
    num_processes = len(gpu_ids)
    chunk_size = math.ceil(len(dataset) / num_processes)
    segments = [dataset[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]
    
    target_key = args.target_key

    # Create a Queue and start processes (unchanged)
    result_queue = Queue()
    processes = []
    for idx, gpu_id in enumerate(gpu_ids):
        p = Process(target=process_segment, args=(segments[idx], args.reward_model, args.eval_mode, gpu_id, result_queue, target_key))
        p.start()
        processes.append(p)

    # Fix: Collect results from the queue BEFORE joining processes
    processed_data = []
    for _ in range(num_processes):  # Expect exactly `num_processes` results
        processed_segment = result_queue.get()  # Blocks until a segment is available
        processed_data.extend(processed_segment)

    # Now wait for all processes to finish (they likely already have)
    for p in processes:
        p.join()
    
    if "completion_reward" in processed_data[0].keys() and f"mistral_dpo_{args.stage}_reward" in processed_data[0].keys():
        for data in processed_data:
            import copy
            temp = copy.deepcopy(data["completion"])
            if data[f"mistral_dpo_{args.stage}_reward"] > data["completion_reward"]:
                data["rejected"] = temp
                data["completion"] = data[f"mistral_dpo_{args.stage}"]
            else:
                data["completion"] = temp
                data["rejected"] = data[f"mistral_dpo_{args.stage}"]

    # Save the updated dataset (unchanged)
    with open(args.save_path, "w", encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/jonghoon/AI-TA/datasets/final/train_dpo_0_test.json")
    parser.add_argument("--save_path", type=str, default="/home/ubuntu/jonghoon/AI-TA/datasets/final/train_dpo_0_test_1.json")
    parser.add_argument("--reward_model", type=str, default="mistralai/Codestral-22B-v0.1")
    parser.add_argument("--target_key", type=str, default="completion")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--eval_mode", type=str, default="initial")
    args = parser.parse_args()
    main(args)
