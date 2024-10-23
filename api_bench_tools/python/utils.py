import queue
import logging
import multiprocessing as mp

from typing import List, Optional, Union
from dataclasses import dataclass, field
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class Request:
    prompt: str
    input_len: int
    output_len: int


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    api_key: Optional[str] = None
    model: Optional[str] = None
    best_of: int = 1
    use_beam_search: bool = False
    client_id: Optional[int] = None
    request_id: int = 0
    num_requests: int = 1


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    output_len: int = 0
    error: str = ""
    client_id: Optional[int] = None
    request_id: int = 0


class IterQueue(queue.Queue):
    def __init__(self, request_list: List[Request]):
        super(IterQueue, self).__init__()
        self.queue = queue.deque(request_list)

    def __iter__(self):
        while True:
            try:
                yield self.get_nowait()
            except queue.Empty:
                break


class IterMpQueue:
    def __init__(self, request_list: List[Request]):
        # Create an internal instance of mp.Queue
        self.mp_queue = mp.Queue()
        # Pre-fill the multiprocessing queue if necessary
        for request in request_list:
            self.mp_queue.put(request)

    def __iter__(self):
        while True:
            try:
                yield self.mp_queue.get_nowait()
            except queue.Empty:
                break

    def empty(self) -> bool:
        return self.mp_queue.empty()

    def clear(self):
        while self.mp_queue.qsize() > 0:
            try:
                self.mp_queue.get_nowait()
            except:
                pass
        
        


def get_tokenizer(
    pretrained_model_name_or_path: str, trust_remote_code: bool
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
    )


def api_url_standardize(base_url: str, endpoint: str, backend: str) -> str:
    """standardize api url

    Args:
        api_url (str): api url

    Returns:
        str: standardized api url
    """
    logger = logging.getLogger()
    match backend:
        case "vllm" | "openai":
            api_url = f"{base_url}{endpoint}"
            if not api_url.startswith("http"):
                api_url = f"http://{api_url}"
        case "ppl":
            api_url = base_url
        case "trt":
            api_url = base_url
            if not api_url.startswith("http"):
                api_url = f"http://{api_url}"
            if not api_url.endswith("/v2/models/ensemble/generate_stream"):
                api_url = f"{api_url}/v2/models/ensemble/generate_stream"
        case "amsv2":
            api_url = base_url
            if not api_url.startswith("http"):
                api_url = f"http://{api_url}"
            if not api_url.endswith("/text-generation/generate_stream"):
                api_url = f"{api_url}/text-generation/generate_stream"
        case "sglang":
            api_url = base_url
            if not api_url.startswith("http"):
                api_url = f"http://{api_url}"
            if not api_url.endswith("/generate"):
                api_url = f"{api_url}/generate"
        case _:
            raise ValueError(f"Unknown backend: {backend}")

    logger.info(f"using {backend} backend with api url: {api_url}")
    return api_url
