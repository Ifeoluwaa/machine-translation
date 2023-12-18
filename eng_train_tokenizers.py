from eng_tokenizer import BytePairTokenizer, BBPETokenizer
import argparse


# CLI parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("dataset_path", type=str, help="Path to dattaset you want to train tokenizer on")
parser.add_argument("source_vocab_size", type=int, help="Vocabulary size.")
#parser.add_argument("target_vocab_size", type=int, help="Vocabulary size.")
parser.add_argument("--eng_tokenizer", type=str, default="bpe", help="Choose tokenization scheme. Write bbpe for byte-level BPE or bpe for vanilla BPE")
parser.add_argument("--out", type=str, default="eng_tokenizer.json", help="output path to save the tokenizer after training")

opt = parser.parse_args()

# Create tokenizer
if opt.eng_tokenizer == "bpe":
    eng_tokenizer = BytePairTokenizer(opt.source_vocab_size)
elif opt.tokenizer == "bbpe":
    eng_tokenizer = BBPETokenizer(opt.source_vocab_size)
else:
    print("Enter a valid tokenizer.")

# Train tokenizer and save as tokenizer.json
eng_tokenizer.train(opt.dataset_path, opt.out, column_name='english')