import time
import torch
import lightseq.inference as lsi
from transformers import BartTokenizer, BartForConditionalGeneration
import sys

BEAM_SIZE = int(sys.argv[1])
print(f"use beam_size {BEAM_SIZE}")

def ls_bart(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def hf_bart(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.generate(inputs, max_length=50, num_beams=BEAM_SIZE)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def main():
    print("initializing bart tokenizer...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    print("creating lightseq model...")
    print(f"reading lightseq_bart_large_{BEAM_SIZE}beam.hdf5")
    ls_model = lsi.Transformer(f"lightseq_bart_large_{BEAM_SIZE}beam.hdf5", 128)
    print("creating huggingface model...")
    hf_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    # GPU warm up
    sentences = [" ".join(["I"] * 10)] * 8
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, pad_to_multiple_of=8)
    inputs_id = inputs["input_ids"]
    _, _ = ls_bart(ls_model, inputs_id)
    _, _ = hf_bart(hf_model, inputs_id)

    bsz_list = [1, 2, 4, 8, 16, 32, 64, 128]
    seq_len_list = [1, 2, 4, 8, 16, 32]
    for bsz in bsz_list:
        total_ls = 0.0
        total_hf = 0.0
        for seq_len in seq_len_list:
            sentences = [" ".join(["I"] * seq_len)] * bsz
            inputs = tokenizer(sentences, return_tensors="pt", padding=True, pad_to_multiple_of=8)
            inputs_id = inputs["input_ids"]
            _, ls_time = ls_bart(ls_model, inputs_id)
            _, hf_time = hf_bart(hf_model, inputs_id)
            total_ls += ls_time
            total_hf += hf_time
        print(f"{bsz}: {total_hf/total_ls-1}")


if __name__ == "__main__":
    main()
