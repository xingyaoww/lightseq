import argparse

parser = argparse.ArgumentParser()
parser.add_argument("output_path")
parser.add_argument("batch_size", type=int)
parser.add_argument("batch_seq_len", type=int)
args = parser.parse_args()

with open(args.output_path, "w") as f:
    f.write(f"{args.batch_size} {args.batch_seq_len}\n")
    for _ in range(args.batch_size):
        f.write("0")
        for _ in range(args.batch_seq_len - 1):
            f.write(" 1")
        f.write("\n")
print(
    f"file generated for batch_size {args.batch_size} and batch_seq_len {args.batch_seq_len} to {args.output_path}."
)
