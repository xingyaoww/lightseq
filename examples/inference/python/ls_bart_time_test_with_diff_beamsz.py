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

def main():
    print("initializing bart tokenizer...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    print("creating lightseq model...")
    weightname = f"lightseq_bart_base_{BEAM_SIZE}beam_embpad.hdf5"
    print(f"reading {weightname}")
    ls_model = lsi.Transformer(weightname, 128)

    sentences = []
    for _ in range(128//4):
        sentences += [
        "I love that girl, but <mask> does not <mask> me.",
        "She is so <mask> that I can not help glance at <mask>.",
        "Nothing's gonna <mask> my love for you.",
        "Drop everything now. Meet me in the pouring <mask>. Kiss me on the sidewalk.",
        ]
    # create batchsize of 128 for profiling
    print("tokenizing the sentences...")
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, 
        # pad_to_multiple_of=8
    )
    inputs_id = inputs["input_ids"]
    
    # GPU warm up
    for _ in range(100):
        _, _ = ls_bart(ls_model, inputs_id)

    # Actual test
    bsz_list = [1, 2, 4, 8, 16, 32, 64, 128]
    for bsz in bsz_list:
        # run this for 10 times for average
        seq_len = [32, 64]
        ls_time = 0
        for _ in range(50):
            _, cur_ls_time = ls_bart(ls_model, inputs_id[:bsz])
            ls_time += cur_ls_time
        ls_time /= 50
        print(f"{bsz}: {ls_time}")


if __name__ == "__main__":
    main()
