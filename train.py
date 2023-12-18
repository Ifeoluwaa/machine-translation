import pandas as pd
from transformers import AutoTokenizer
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import DataLoader, Dataset
from preprocessing import clean_data, train_test_split
from preprocessing import CharacterTokenizer, read_and_preprocess_data
from config import Config
from learning_schedule import LearningRateScheduler
from transformer import Transformer
from eng_tokenizer import eng_load_tokenizer
from yor_tokenizer import yor_load_tokenizer
from learning_schedule import LearningRateScheduler
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_AMP = True  # Use Gradient scaler and mixed precision
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CorpusDataset(Dataset):
    def __init__(self, data, source_vocab_size, target_vocab_size):
        self.data = data
        self.n_data = len(data)
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        X = torch.LongTensor(self.data[idx: idx + self.source_vocab_size])
        y = torch.LongTensor(self.data[idx + 1: idx + 1 + self.target_vocab_size])

        return X, y

class NaiveDataLoader:
    def __init__(self, data, source_vocab_size, target_vocab_size, batch_size):
        self.data = data
        self.n_data = len(data)
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.n_batches = int(self.n_data / self.batch_size)

    def __len__(self):
        return self.batch_size

    def get_batch(self):
        for _ in range(self.n_batches):
            idx1 = torch.randint(len(self.data) - self.source_vocab_size, (self.batch_size,))
            idx2 = torch.randint(len(self.data) - self.target_vocab_size, (self.batch_size,))
            X = torch.LongTensor([self.data[i: i+self.source_vocab_size] for i in idx1])
            y = torch.LongTensor([self.data[i+1: i+1+self.target_vocab_size] for i in idx2])

            yield X, y


def evaluate(model, X_val, y_val, criterion):
    with torch.no_grad():
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            y_pred = model(X_val)
            val_loss = criterion(y_pred.view(-1, y_pred.size(-1)), y_val.view(-1))
            return val_loss

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(opt.emb_dim, opt.source_vocab_size, opt.target_vocab_size, 
                        opt.source_seq_len, opt.target_seq_len, opt.input_dimension, 
                        opt.num_heads, opt.num_layers).to(device)

    datasets = read_and_preprocess_data(opt.dataset_path)

    # Split the dataset
    train, test = train_test_split(datasets, train_size=0.8)

    # Load tokenizers
    eng_bpe_tokenizer = eng_load_tokenizer(opt.eng_tokenizer)
    yor_bpe_tokenizer = yor_load_tokenizer(opt.yor_tokenizer)

    # Tokenize and encode
    eng_train_token_ids = CharacterTokenizer.tokenize_and_encode(train, eng_bpe_tokenizer)
    eng_test_token_ids = CharacterTokenizer.tokenize_and_encode(test, eng_bpe_tokenizer)
    
    yor_train_token_ids = CharacterTokenizer.tokenize_and_encode(train, yor_bpe_tokenizer)
    yor_test_token_ids = CharacterTokenizer.tokenize_and_encode(test, yor_bpe_tokenizer)

    # Create data loaders
    eng_train_dataloader = NaiveDataLoader(eng_train_token_ids, source_vocab_size=opt.source_vocab_size, batch_size=opt.batch_size)
    eng_test_dataloader = NaiveDataLoader(eng_test_token_ids, source_vocab_size=opt.source_vocab_size, batch_size=opt.batch_size)
    
    yor_train_dataloader = NaiveDataLoader(yor_train_token_ids, target_vocab_size=opt.target_vocab_size, batch_size=opt.batch_size)
    yor_test_dataloader = NaiveDataLoader(yor_test_token_ids, target_vocab_size=opt.target_vocab_size, batch_size=opt.batch_size)
    
    eng_train_iter = iter(eng_train_dataloader.get_batch())
    yor_train_iter = iter(yor_train_dataloader.get_batch())

    eng_max_iters = int(len(eng_train_token_ids) / opt.batch_size)
    yor_max_iters = int(len(yor_train_token_ids) / opt.batch_size)

    # Learning rate scheduler
    lr_scheduler = LearningRateScheduler(max_lr=2.4e-4, min_lr=0.0, warm_up_iters=2000, max_iters=eng_max_iters)
    learning_rate = lr_scheduler.get_lr(0)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # GradScaler and Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    for i, (source, target) in enumerate(zip(eng_train_iter, yor_train_iter)):
        source = source.to(device)
        target = target.to(device)
            
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
            target_pred = model(source, target)
            loss = criterion(target_pred.view(-1, target_pred.size(-1)), target.view(-1))
            loss = loss / opt.num_accumulation_steps

            # Accumulates scaled gradients
            scaler.scale(loss).backward()

        if (i + 1) % opt.num_accumulation_steps == 0:
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Evaluation
        if i % opt.epochs_log == 0:
            model.eval()
            X_val, y_val = next(zip(eng_test_dataloader.get_batch(), yor_test_dataloader.get_batch()))
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            val_loss = evaluate(model, X_val, y_val, criterion=criterion)
            print(f"At epoch {i}, Train loss: {loss.item()}      Val loss: {val_loss.item()}")
            
            # Save model checkpoint only if validation loss improves
            if i == 0 or val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving model checkpoint at epoch {i}...")
                torch.save({
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "Optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss
                }, "model_checkpoint.tar")
                print("Model checkpoint saved as model_checkpoint.tar")

            model.train()
        
        print(f"Train loss: {loss.item()}")

if __name__ == "__main__":
    opt = Config().parse()
    best_val_loss = float('inf')  # Initialize with a high




