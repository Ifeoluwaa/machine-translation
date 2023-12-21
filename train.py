import pandas as pd
from transformer import AutoTokenizer
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
from torch.cuda.amp import GradScaler

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_AMP = True  # Use Gradient scaler and mixed precision
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CorpusDataset(Dataset):
    def __init__(self, data, vocab_size):
        self.data = data
        self.n_data = len(data)
        self.vocab_size = vocab_size

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        X = torch.LongTensor(self.data[idx: idx + self.vocab_size])
        y = torch.LongTensor(self.data[idx + 1: idx + 1 + self.vocab_size])

        return X, y

class NaiveDataLoader:
    def __init__(self, data, vocab_size, batch_size):
        self.data = data
        self.n_data = len(data)
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.n_batches = int(self.n_data / self.batch_size)

    def __len__(self):
        return self.batch_size

    def get_batch(self):
        for _ in range(self.n_batches):
            idx1 = torch.randint(len(self.data) - self.vocab_size, (self.batch_size,))
            idx2 = torch.randint(len(self.data) - self.vocab_size, (self.batch_size,))
            X = torch.stack([torch.LongTensor(self.data[i: i+self.vocab_size]) for i in idx1])
            y = torch.stack([torch.LongTensor(self.data[i+1: i+1+self.vocab_size]) for i in idx2])

            yield X, y

def evaluate(model, X_val, y_val, criterion, device):
    with torch.no_grad():
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        with torch.cuda.amp.autocast(enabled=True):  # Enable autocast
            y_pred = model(X_val)
            val_loss = criterion(y_pred.view(-1, y_pred.size(-1)), y_val.view(-1))
            return val_loss

def main(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(opt.vocab_size, opt.emb_dim, opt.num_layers, opt.seq_len, opt.num_heads).to(device)
    
    # Assuming these functions are defined in your utils.py file
    eng_bpe_tokenizer = eng_load_tokenizer(opt.eng_tokenizer)
    yor_bpe_tokenizer = yor_load_tokenizer(opt.yor_tokenizer)

    datasets = read_and_preprocess_data(opt.dataset_path)
    train, test = train_test_split(datasets, train_size=0.8)

    eng_train_token_ids = CharacterTokenizer.tokenize_and_encode(train, eng_bpe_tokenizer)
    yor_train_token_ids = CharacterTokenizer.tokenize_and_encode(train, yor_bpe_tokenizer)

    eng_train_dataloader = NaiveDataLoader(eng_train_token_ids, vocab_size=opt.vocab_size, batch_size=opt.batch_size)
    yor_train_dataloader = NaiveDataLoader(yor_train_token_ids, vocab_size=opt.vocab_size, batch_size=opt.batch_size)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    # GradScaler and Mixed precision
    scaler = GradScaler(enabled=getattr(opt, 'use_amp', False))

    best_val_loss = float('inf')  # Initialize with a high

    for epoch in range(opt.EPOCHS):
        model.train()

        eng_train_iter = iter(eng_train_dataloader.get_batch())
        yor_train_iter = iter(yor_train_dataloader.get_batch())

        for j, ((eng_source, eng_target), (yor_source, yor_target)) in enumerate(zip(eng_train_iter, yor_train_iter)):
            eng_source = eng_source.to(device)
            eng_target = eng_target.to(device)
            yor_source = yor_source.to(device)
            yor_target = yor_target.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                target_pred = model(eng_source, yor_target)
                loss = criterion(target_pred.view(-1, target_pred.size(-1)), yor_target.view(-1))
                loss = loss / opt.num_accumulation_steps

                # Accumulates scaled gradients
                scaler.scale(loss).backward()

            if (j + 1) % opt.num_accumulation_steps == 0:
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Print or log training loss
                if (j + 1) % opt.print_every == 0:
                    print(f"Epoch {epoch+1}, Batch {j+1}, Train loss: {loss.item()}")

        model.eval()

        # Use the validation data loader to get batches for evaluation
        X_val, y_val = next(zip(eng_train_dataloader.get_batch(), yor_train_dataloader.get_batch()))
        X_val = X_val.to(device)  # Move validation data to the same device
        y_val = y_val.to(device)
        val_loss = evaluate(model, X_val, y_val, criterion=criterion, device=device)
        print(f"At epoch {epoch+1}, Train loss: {loss.item()}      Val loss: {val_loss.item()}")

        # Save model checkpoint only if validation loss improves
        if epoch == 0 or val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving model checkpoint at epoch {epoch+1}...")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "Optimizer_state_dict": optimizer.state_dict(),
                "loss": loss
            }, "model_checkpoint.tar")
            print("Model checkpoint saved as model_checkpoint.tar")

if __name__ == "__main__":
    opt = Config().parse()
    main(opt)