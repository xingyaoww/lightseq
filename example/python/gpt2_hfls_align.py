import os
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import lightseq

ls_gpt2 = lightseq.Gpt("lightseq_gpt2.pb", max_batch_size=16, max_step=50)
hf_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
hf_gpt2.to('cuda:0')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# add EOS as PAD to avoid warning according to https://huggingface.co/blog/how-to-generate
tokenizer.pad_token = tokenizer.eos_token_id

sentences = ["I love that girl, but"]
inputs = tokenizer(sentences, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"]  # .cuda()
print(input_ids)

# huggingface - greedy
hf_res = hf_gpt2.generate(
    input_ids.to('cuda:0'),
    max_length=50,
    # output_scores=True,
    # return_dict_in_generate=True,
)
# print(f"logits: {torch.cat(hf_res.scores).flatten().detach().cpu().numpy()}")

# huggingface - sampling top1
hf_res_dosample = hf_gpt2.generate(
    input_ids.to('cuda:0'),
    max_length=50,
    do_sample=True,
    top_k=1,
)

# lightseq - sampling top1
ls_res = ls_gpt2.sample(
    input_ids,
    sampling_method="topk",
    topk=1,
    topp=1
)

print("======== huggingface ========")
print("hf - greedy top1 results:")
print(f"res: {hf_res}")
print(tokenizer.batch_decode(hf_res, skip_special_tokens=True))
print("hf - do sample results:")
print(f"res: {hf_res_dosample}")
print(tokenizer.batch_decode(hf_res_dosample, skip_special_tokens=True))
print("======== lightseq ========")
print(f"res: {ls_res}")
print(tokenizer.batch_decode(ls_res, skip_special_tokens=True))
