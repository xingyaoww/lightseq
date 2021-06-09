import time
import argparse

import torch
import numpy as np
import lightseq
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def ls_gpt2(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.sample(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time-start_time


def hf_gpt2(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.generate(
        inputs["input_ids"].to('cuda:0'),
        max_length=50,
    )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time-start_time


def ls_generate(model, tokenizer, inputs):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_res_ids, ls_time = ls_gpt2(model, inputs["input_ids"])
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    for sent in ls_res:
        print(sent)
        print("------")
    print(ls_res_ids)


def hf_generate(model, tokenizer, inputs):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_res_ids, hf_time = hf_gpt2(model, inputs)
    hf_res = tokenizer.batch_decode(hf_res_ids, skip_special_tokens=True)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    for sent in hf_res:
        print(sent)
        print("------")
    print(hf_res_ids)


def warmup(ls_tokenizer, hf_tokenizer, ls_model, hf_model, sentences):
    ls_inputs = ls_tokenizer(sentences, return_tensors="pt", padding=True)
    # ls_input_ids = ls_inputs["input_ids"]

    hf_inputs = hf_tokenizer(sentences, return_tensors="pt", padding=True)
    # hf_input_ids = hf_inputs["input_ids"]

    ls_generate(ls_model, ls_tokenizer, ls_inputs)
    hf_generate(hf_model, hf_tokenizer, hf_inputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_input', action="store_true")
    args = parser.parse_args()

    print("initializing gpt tokenizer...")

    ls_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # lightseq use len(tokenizer) as pad_token in default
    ls_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"lightseq tokenizer pad token id: {ls_tokenizer.pad_token_id}")

    hf_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # use EOS as PAD for huggingface to avoid warning according to https://huggingface.co/blog/how-to-generate while avoid reshaping the model embedding
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    print(f"huggingface tokenizer pad token id: {hf_tokenizer.pad_token_id}")

    print("creating lightseq model...")
    ls_model = lightseq.Gpt(
        "lightseq_gpt2.pb",
        max_batch_size=16,
        max_step=50
    )

    print("creating huggingface model...")
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    hf_model.to('cuda:0')

    sentences = [
        "I love that girl, but",
        # "She is so beautiful that I can not help",
        # "Nothing's gonna stop",
        # "Drop everything now. Meet me in the pouring"
    ]

    print("====================START warmup====================")
    warmup(ls_tokenizer, hf_tokenizer, ls_model, hf_model, sentences)
    print("====================END warmup====================")

    while True:
        if args.user_input:
            sentences = [input("input the masked sentence:\n")]

        print("tokenizing the sentences...")

        ls_inputs = ls_tokenizer(sentences, return_tensors="pt", padding=True)

        hf_inputs = hf_tokenizer(sentences, return_tensors="pt", padding=True)

        print(hf_inputs)
        print(ls_inputs)

        ls_generate(ls_model, ls_tokenizer, ls_inputs)
        hf_generate(hf_model, hf_tokenizer, hf_inputs)

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
