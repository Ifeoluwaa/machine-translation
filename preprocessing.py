from typing import List, Tuple
import string
from string import punctuation
import random
import pandas as pd

class CharacterTokenizer:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        self.vocabulary, self.vocab_map = self._create_vocabulary(self.corpus)
        self.vocabulary_length = len(self.vocabulary)

    def _create_token_map(self, tokens: set):
        tokens_map = {}
        for id, token in enumerate(tokens):
            tokens_map[token] = id

        return tokens_map

    def _create_vocabulary(self, corpus: List[str]) -> List[str]:
        # list_of_characters = [word.split() for sentence in corpus for word in sentence]
        characters = []
        for sentence in corpus:
            for character in sentence:
                characters.append(character)
        unique_tokens = set(characters)
        tokens_map = self._create_token_map(unique_tokens)

        return unique_tokens, tokens_map

    
    def tokenize_and_encode(dataset: List[List[str]], tokenizer):
        token_ids = []

        for sentence in dataset:
            if isinstance(sentence, list):
            # If the sentence is a list, join its elements into a single string
                sentence = " ".join(sentence)
            output = tokenizer.encode(sentence)
            token_ids += output.ids

        return token_ids

    
    def decode(self, indexes: List[int]) -> List[str]:
        characters = []
        for id in indexes:
            characters.append(self.vocab_map.get(id, 0))

        return characters
    
def train_test_split(corpus, train_size, shuffle=False):
    if shuffle:
        random.shuffle(corpus)
    train_size = int(train_size * len(corpus))
    train_arr = corpus[:train_size]
    test_arr = corpus[train_size + 1:]
    return train_arr, test_arr


def tokenize_and_encode(dataset: List[str], tokenizer):
    token_ids = []

    for sentence in dataset:
        output = tokenizer.encode(sentence)
        token_ids += output.ids

    return token_ids


def clean_data(sentence):
  punctuations = string.punctuation
  sentence = sentence.translate(str.maketrans("", "", punctuations))
  return sentence



def read_and_preprocess_data(file_path: str = 'eng_yor_data.xlsx') -> Tuple[List[str], List[str]]:
    try:
        # Use openpyxl engine
        data_file = pd.read_excel(file_path, index_col='ID', engine='openpyxl')
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return [], []

    def read_lines(df):
        n_rows = df.shape[0]
        for row in range(n_rows):
            contents = df.iloc[row]
            yield [contents["english"].lower(), contents["yoruba"].lower()]

    eng_corpus = []
    yoruba_corpus = []

    for eng, yor in read_lines(data_file):
        eng_corpus.append(clean_data(eng))
        yoruba_corpus.append(clean_data(yor))

    return eng_corpus, yoruba_corpus

# Usage
eng_corpus, yoruba_corpus = read_and_preprocess_data()
#print(yoruba_corpus)