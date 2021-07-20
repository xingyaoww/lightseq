import time
import argparse

import torch
import lightseq.inference as lsi
from transformers import BartTokenizer, BartForConditionalGeneration


def ls_bart(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.infer(inputs)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time

def ls_generate(model, tokenizer, inputs_id):
    # print("=========lightseq=========")
    # print("lightseq generating...")
    ls_res_ids, ls_time = ls_bart(model, inputs_id)
    ls_res_ids = [ids[0] for ids in ls_res_ids[0]]
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    # print(f"lightseq time: {ls_time}s")
    # print("lightseq results:")
    # for sent in ls_res:
    #     print(sent)

def main():
    print("initializing bart tokenizer...")
    # change to "facebook/bart-base" for base model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    print("creating lightseq model...")
    # change to "lightseq_bart_base.hdf5" for base model
    ls_model = lsi.Transformer("lightseq_bart_base_4beam_embpad.hdf5", 128)

    _sents = [
        "I love that girl, but <mask> does not <mask> me.",
        "She is so <mask> that I can not help glance at <mask>.",
        "Nothing's gonna <mask> my love for you.",
        "Drop everything now. Meet me in the pouring <mask>. Kiss me on the sidewalk.",
    ]
    sentences = []
    for _ in range(128//4):
        sentences += _sents
    # create batchsize of 128 for profiling

    print("tokenizing the sentences...")
    inputs = tokenizer(sentences, 
        return_tensors="pt", 
        padding=True, 
        pad_to_multiple_of=8
    )
    inputs_id = inputs["input_ids"]
    for _ in range(10):
        ls_generate(ls_model, tokenizer, inputs_id)

if __name__ == "__main__":
    main()
