import os
import requests, json

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 16,
        },
        "stream": True,
    },
    stream=True,
)

prev = 0
for chunk in response.iter_lines(decode_unicode=False):
    chunk = chunk.decode("utf-8")
    print(chunk)
    # if chunk and chunk.startswith("data:"):
    #     if chunk == "data: [DONE]":
    #         break
    #     data = json.loads(chunk[5:].strip("\n"))
    #     output = data["text"].strip()
    #     print(output[prev:], end="", flush=True)
    #     prev = len(output)
print("")