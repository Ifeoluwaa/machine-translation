from yor_tokenizer import BytePairTokenizer, BBPETokenizer
import argparse


# CLI parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("dataset_path", type=str, help="Path to dattaset you want to train tokenizer on")
parser.add_argument("target_vocab_size", type=int, help="Vocabulary size.")
parser.add_argument("--yor_tokenizer", type=str, default="bpe", help="Choose tokenization scheme. Write bbpe for byte-level BPE or bpe for vanilla BPE")
parser.add_argument("--out", type=str, default="yor_tokenizer.json", help="output path to save the tokenizer after training")

opt = parser.parse_args()

# Create tokenizer
if opt.yor_tokenizer == "bpe":
    yor_tokenizer = BytePairTokenizer(opt.target_vocab_size)
elif opt.yor_tokenizer == "bbpe":
    yor_tokenizer = BBPETokenizer(opt.target_vocab_size)
else:
    print("Enter a valid tokenizer.")

# Train tokenizer and save as tokenizer.json
yor_tokenizer.train(opt.dataset_path, opt.out, column_name='yoruba')