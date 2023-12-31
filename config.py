import argparse

class ConfigBase:
    def __init__(self):
        self.name = argparse.Namespace()
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialization(self):
        self.parser.add_argument("dataset_path", type=str,
                            help="dataset file. Should be a txt file.")
        
        self.parser.add_argument("--vocab_size", type=int, default=7000,
                            help="Vocabulary size. Vocabulary is created with byte-level byte pair encoding")
        self.parser.add_argument("--learning_rate", type=float, default=1e-5, help="starting learning rate")
        #self.parser.add_argument("--target_vocab_size", type=int, default=7000,
                            #help="Vocabulary size. Vocabulary is created with byte-level byte pair encoding")
        self.parser.add_argument("--seq_len", type=int, default=50, help="Maximum sequence length for the two languages")
        self.parser.add_argument("--use_amp", action="store_true", help="Enable automatic mixed precision training")

        self.parser.add_argument("--dropout_probability", type=float, default=0.1, 
                            help="Dropout probability. Value must be between 0 and 1.")
        self.parser.add_argument("--emb_dim", type=int, default=164,
                            help="Embedding dimension. Make sure this argument is a multiple of attention heads")
        self.parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer blocks")
        self.parser.add_argument("--num_heads", type=int, default=4,
                            help="Number of attention heads. Make sure this argument is a divisor of Embedding dimension")
        self.parser.add_argument("--EPOCHS", type=int, default=2, help="Number of epochs")
        self.parser.add_argument("--epochs_log", type=int, default=1,
                                 help="Logging routine. Epochs_log tells the program to log every epochs_log")
        self.parser.add_argument("--batch_size", type=int, default=10, help="Batch size")
        self.parser.add_argument("--num_accumulation_steps", type=int, default=4, help="Gradient Accumulation steps")
        self.parser.add_argument("--eng_tokenizer", type=str, default="eng_tokenizer.json", help="Path to eng_tokenizer")
        self.parser.add_argument("--yor_tokenizer", type=str, default="yor_tokenizer.json", help="Path to yor_tokenizer")

    def _parse(self):
        self.opt = self.parser.parse_args()
        return self.opt


class Config(ConfigBase):
    def __init__(self):
        super().__init__()
        self.initialization()

    def parse(self):
        opt = self._parse()

        return opt