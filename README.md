# Adaptive Huffman and Recurrent Neural Network Combination for Text Lossless Compression
A final project in information theory course.
In Order to compress a text file, we suggest training a character-level LSTM language model,
and use the output probabilities to build an adaptive Huffman tree, based on the character context.

## Requirements
Experiments are done using PyTorch.

## Usage
for compression run
```commandline
python main.py compress file.txt
```
and for decompression run
```commandline
python main.py decompress compressed_file.bin
```
#### Relevant arguments:
- `--criterion`: Loss function to use for model training. `CE` for Cross-Entropy loss,
`L1` for L1 loss, `L2` for L2 loss (Mean Squared Error)
- `--threshold`: Threshold used when pruning the adaptive Huffman tree (see the PDF file).
If an integer is given, the threshold is the number of leafs in the tree.
In a float is given, the tree contains character probabilities above the threshold.  
- `--limit`: Our algorithm is very slow and cannot decompress large file in reasonable time.
This parameter sets the number of characters to compress or decompress. We suggest using `--limit=10000`
for compressing the first 10000 characters in the text file.
- `--model_file`: When compressing, a LSTM model is trained and saved in a file 
(Regardless the compressed file which also includes the model). The training process might take some time.
For compressing experiments it is better using the already trained model with this argument.
It this argument is not supplied, a LSTM model will be trained before compression.

#### LSTM model training arguments:
- `--emsize` size of character embeddings
- `--nhid` number of hidden units per LSTM layer
- `--nlayers` number of LSTM layers
- `--lr` initial learning rate
- `--epochs` number of train epochs (default is 1)
- `--dropout` dropout applied to layers
- `--tied` tie the word embedding and softmax weights
- `--log-interval` report interval
- `--save` path to save the final model

## Code References
1. [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model)
1. [Huffman Coding](https://bhrigu.me/blog/2017/01/17/huffman-coding-python-implementation/)
