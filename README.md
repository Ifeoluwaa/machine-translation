**Transfor-Translator**

This is a machine translation project using transformer introduced in 'Attention is all you need'. The dataset used for this experiment is English-Yoruba and it's not a large dataset. You can use train.py file to train your own machine translation model.

**Installations**

Dependencies:
1. [pytorch]
2. [numpy]
3. [pip install tokenizers] for huggingface tokenizers

**How to run**

There is a config.py where all the arguments are listed. These arguments are used in the train.py where training takes place.

**Tokenizer**

The dataset has two corpora; one for 'english' and the other for 'yoruba'. The eng_trained_tokenizers.py trained the english corpus while the yor_trained_tokenizers.py trained the yoruba corpus. The BPE in the trained tokenizers was used to tokenize. 

**Training**
It is advisable to use GPU.

[$ !python train.py "/content/drive/MyDrive/eng_yor_data.xlsx" --vocab_size 7000 --seq_len 50 --emb_dim 256 --num_layers 4 --num_heads 4 --EPOCHS 2 --epochs_log 1 --batch_size 16 --num_accumulation_steps 4 --eng_tokenizer=eng_tokenizer.json --yor_tokenizer=yor_tokenizer.json]

