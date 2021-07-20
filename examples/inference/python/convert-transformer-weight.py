import numpy as np
import tensorflow as tf
import argparse
from transformer_pb2 import Transformer

parser = argparse.ArgumentParser(
    description="Change certain parameter of an existing Transformer proto file."
)
parser.add_argument("input_weight_file")
parser.add_argument("output_weight_file")
parser.add_argument("--beam_size", type=int)
parser.add_argument("--extra_decode_length", type=int)
parser.add_argument("--sampling_method", type=str)
parser.add_argument("--topp", type=float)
parser.add_argument("--topk", type=int)
parser.add_argument("--pad_token_embedding", action="store_true")

args = parser.parse_args()
print(args)
ARG_LIST = ["beam_size", "extra_decode_length", "sampling_method", "topp", "topk"]
if args.input_weight_file.endswith(".pb"):
    print(f"processing weight file: {args.input_weight_file}")
    transformer = Transformer()
    with tf.io.gfile.GFile(args.input_weight_file, "rb") as fin:
        transformer.ParseFromString(fin.read())

    print("==== before convertion ====")
    print(transformer.model_conf)

    print("==== configuration update ====")
    for _cur_arg in ARG_LIST:
        if getattr(args, _cur_arg) is not None:
            _cur_arg_val = getattr(args, _cur_arg)
            setattr(transformer.model_conf, _cur_arg, _cur_arg_val)
            print(f"{_cur_arg} is updated to {_cur_arg_val}")

    print("==== after convertion ====")
    print(transformer.model_conf)

    if args.pad_token_embedding:
        print(f"enable token embedding padding for Tensor Core performance gain.")

        # 1. pad token embedding for src_embedding
        _enc_hidden_size = len(transformer.src_embedding.norm_scale[:])
        # encoder token embedding: [src_vocab_size, hidden_size]
        _enc_token_embedding = np.array(
            transformer.src_embedding.token_embedding[:]
        ).reshape((-1, _enc_hidden_size))
        _enc_vocab_size = _enc_token_embedding.shape[0]
        print(
            f"encoder token embedding parsed with shape: {_enc_token_embedding.shape}"
        )

        if _enc_vocab_size % 8 != 0:
            _pad_amt = ((_enc_vocab_size // 8) + 1) * 8 - _enc_vocab_size
            print(
                f"encoder token embedding shape is not multiple of 8, pad amount {_pad_amt} for TensorCore performance."
            )
            assert _pad_amt > 0
            _enc_token_embedding = np.pad(
                _enc_token_embedding,
                ((0, _pad_amt), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            transformer.src_embedding.token_embedding[
                :
            ] = _enc_token_embedding.flatten().tolist()
            print(
                f"transformer.src_embedding.token_embedding padded to shape {_enc_token_embedding.shape}"
            )

        # 2. pad token embedding & shared bias of trg_embedding
        _dec_hidden_size = len(transformer.trg_embedding.norm_scale[:])
        # decoder token embedding: [hidden_size, trg_vocab_size]
        _dec_token_embedding = np.array(
            transformer.trg_embedding.token_embedding[:]
        ).reshape((_dec_hidden_size, -1))
        _dec_vocab_size = _dec_token_embedding.shape[1]
        print(
            f"decoder token embedding parsed with shape: {_dec_token_embedding.shape}"
        )

        if _dec_vocab_size % 8 != 0:
            _pad_amt = ((_dec_vocab_size // 8) + 1) * 8 - _dec_vocab_size
            print(
                f"decoder token embedding shape is not multiple of 8, pad amount {_pad_amt} for TensorCore performance."
            )
            assert _pad_amt > 0
            _dec_token_embedding = np.pad(
                _dec_token_embedding,
                ((0, 0), (0, _pad_amt)),
                mode="constant",
                constant_values=0,
            )
            transformer.trg_embedding.token_embedding[
                :
            ] = _dec_token_embedding.flatten().tolist()
            print(
                f"transformer.trg_embedding.token_embedding padded to shape {_dec_token_embedding.shape}"
            )

            # pad bias for token_embedding too
            shared_bias = np.array(transformer.trg_embedding.shared_bias[:])
            shared_bias = np.pad(
                shared_bias, ((0, _pad_amt)), mode="constant", constant_values=0
            )
            transformer.trg_embedding.shared_bias[:] = shared_bias.flatten().tolist()
            print(
                f"transformer.trg_embedding.shared_bias padded to shape {shared_bias.shape}"
            )

    print(f"writing processed weight file to {args.output_weight_file}")
    with tf.io.gfile.GFile(args.output_weight_file, "wb") as fout:
        fout.write(transformer.SerializeToString())
else:
    assert False, f"Extension type of file {args.input_weight_file} is not supported."
