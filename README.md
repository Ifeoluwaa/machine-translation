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

[$ python train.py dataset_path --source_vocab_size=source_vocab_size --target_vocab_size=target_vocab_size --source_seq_len --target_seq_len --emb_dim=256 --num_layers=4 --num_heads=4 --EPOCHS=2 --input_dimension=8 --epochs_log --batch_size --num_accumulation_steps --tokenizer=path-to-tokenizer]

