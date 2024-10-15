import json
import random
from dataclasses import dataclass
from typing import AsyncGenerator, List, Optional, Tuple
from transformers import PreTrainedTokenizerBase, AutoTokenizer


@dataclass
class Request:
    prompt: str
    input_len: int
    output_len: int


def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    num_turns: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
    system_prompt_path: Optional[str] = None,
) -> List[Request]:
    """sample sharegpt format dataset to requests.

    Args:
        dataset_path (str): path to the dataset.
        num_requests (int): number of requests to sample.
        num_turns (int): number of conversation turns in each request. One turn is a prompt and a completion.
        tokenizer (PreTrainedTokenizerBase): transformers tokenizer
        fixed_output_len (Optional[int], optional): Fixed output length can be set. Defaults to None.
        system_prompt_path (Optional[str], optional): System prompt path. Defaults to None.

    Raises:
        ValueError: output_len too small

    Returns:
        List[Request]: filtered dataset, each element is a Request instance.
    """
    # print("[I] Sampling requests...")
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    num_turns *= 2 # Each turn has a prompt and a completion.
    # Filter out the conversations with less than num_turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= num_turns]
    # Only keep the first num_turns of each conversation.
    dataset = [[data["conversations"][turn]["value"] for turn in range(num_turns)] for data in dataset]


    # Shuffle the dataset.
    random.seed(0)
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Request] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break
        
        prompt = ""
        for j in range(num_turns - 1):
            prompt += dataset[i][j] + "\n"
        completion = dataset[i][-1]
        
        if system_prompt_path is not None:
            with open(system_prompt_path) as f:
                prompt = f.read() + '\n' + prompt
            
        
        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(prompt).input_ids
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        
        filtered_dataset.append(Request(prompt, prompt_len, output_len))
        
        if i == len(dataset) - 1:
            i = 0

    return filtered_dataset


def sample_xiaomi_requests(
    dataset_path: str,
    num_requests: int,
    num_turns: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: int = 200,
    system_prompt_path: Optional[str] = None,
) -> List[Tuple[str, int, int]]:
    """sample sharegpt format dataset to requests.

    Args:
        dataset_path (str): path to the dataset.
        num_requests (int): number of requests to sample.
        num_turns (int): number of conversation turns in each request. One turn is a prompt and a completion.
        tokenizer (PreTrainedTokenizerBase): transformers tokenizer
        fixed_output_len (Optional[int], optional): Fixed output length can be set. Defaults to None.
        system_prompt_path (Optional[str], optional): System prompt path. Defaults to None.

    Raises:
        ValueError: output_len too small

    Returns:
        List[Tuple[str, int, int]]: filtered dataset, each element is a tuple of (prompt, input_len, output_len)
    """
    # print("[I] Sampling requests...")
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f.readlines()]
        
    dataset = [[message["content"] for message in data["messages"]] for data in dataset]
    dataset = ["\n".join(data) for data in dataset]
    
    # Shuffle the dataset.
    random.seed(0)
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break
        
        prompt = dataset[i]
        
        if system_prompt_path is not None:
            with open(system_prompt_path) as f:
                prompt = f.read() + '\n' + prompt
        
        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(prompt).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        
        filtered_dataset.append((prompt, prompt_len, output_len))
        
        if i == len(dataset) - 1:
            i = 0

    return filtered_dataset


DATASET_SAMPLE = {
    "sharegpt": sample_sharegpt_requests,
    "xiaomi": sample_xiaomi_requests,
}


if __name__ == "__main__":
    requests = sample_xiaomi_requests(
        dataset_path="/mnt/llm/workspace/zhaozhiyu/work/llm.perf/api_bench_tools/datasets/xiaomi_data1_medium.jsonl",
        num_requests=100,
        num_turns=2,
        tokenizer=AutoTokenizer.from_pretrained("/mnt/llm2/llm_perf/hf_models/llama-7b-hf"),
    )
    
    print(requests[0])