from tokenizers import pre_tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation
from tokenizers.normalizers import Lowercase
from time import sleep
from typing import List
import pandas as pd

class BytePairTokenizer:
    def __init__(self, source_vocab_size: int, special_tokens="[UNK]") -> None:
        self.normalizer = Lowercase()
        self.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits(), Punctuation()])
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.trainer = BpeTrainer(source_vocab_size=source_vocab_size, show_progress=True, special_tokens=[special_tokens])
        self.tokenizer.normalizer = self.normalizer
        self.tokenizer.pre_tokenizer = self.pre_tokenizer

    def train(self, file_path: str, save_to: str, column_name: str) -> None:
        print("Training tokenizer...")
        data_file = pd.read_excel(file_path, index_col='ID', engine='openpyxl')
        yor_corpus = data_file[column_name].tolist()

        # Tokenize only the English part
        self.tokenizer.train_from_iterator(yor_corpus, trainer=self.trainer)
        
        print("Training complete.")
        print(f"Saving tokenizer as {save_to}...")
        self.tokenizer.save(save_to)
        sleep(2)
        print("Tokenizer saved successfully!")

# Usage
#tokenizer = BytePairTokenizer(source_vocab_size=your_source_vocab_size)
#tokenizer.train(file_path='eng_yor_data.xlsx', save_to='english_tokenizer.json', column_name='english')

class BBPETokenizer:
    def __init__(self, target_vocab_size: int, special_tokens=["<|endoftext|>"]):
        self.tokenizer = Tokenizer(BPE())
        self.trainer = BpeTrainer(target_vocab_size=target_vocab_size, show_progress=True, special_tokens=special_tokens)
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    def train(self, file_path: str, save_to: str, column_name: str) -> None:
        print("Training tokenizer...")
        data_file = pd.read_excel(file_path, index_col='ID', engine='openpyxl')
        yor_corpus = data_file[column_name].tolist()
        
        self.tokenizer.train_from_iterator(yor_corpus, trainer=self.trainer)
        
        print("Training complete.")
        print(f"Saving tokenizer as {save_to}...")
        self.tokenizer.save(save_to)
        sleep(2)
        print("Tokenizer saved successfully!")

def yor_load_tokenizer(path: str):
    #print("Loading tokenizer from:", path)
    return Tokenizer.from_file(path)


import os

file_path = 'eng_yor_data.xlsx'
if not os.path.exists(file_path):
    print(f"Error: File not found - {file_path}")

