import numpy as np
import model
import train
import os
import heapq
import collections
import operator
import ast
import sys
import time
from data import Corpus
import torch
import argparse
import torch.nn.functional as F

# todo remove transformer code
# todo char batches?
# todo start symbol in train time (or not? check init_hidden)
# todo threshold in huffman dictionary before building the tree

# read function !DROR! to include '.' and others as words.
def read_data_dror(fname):
    data = []
    vec = []
    text = []
    count = len(open(fname).readlines())
    idxl = 0
    # with open(fname, 'rb') as f:
    #     file = f.read()
    with open(fname, encoding="utf-8", errors='ignore') as f:
        for line in file:
            idxl = idxl+1
            if line.strip().split():
                vec = vec + line.strip().split()
                text = text + line.strip().split()
            else:
                data.append(vec)
                vec = []
            print(100*(idxl/count))
    vocab = set(text)
    return data, vocab


def read_data(fname):
    data = []
    vec = []
    text = []
    count = 0
    # with open(fname, 'rb') as f:
    #     file = f.read()
    #     file = file.decode('utf-8', 'ignore')
    #     for line in file:
    #         count += 1
    #         # if count < 50:
    #         #     print(line)
    with open(fname, encoding="ascii", errors="surrogateescape") as file:
        for line in file:
            count += 1
    print('number of lines: ', count)
    i = 0
    with open(fname, encoding="ascii", errors="surrogateescape") as file:
        for line in file:
            pct = int((i / count) * 100)
            if i < 5:
                print(line)
            i += 1
            if line.strip().split():
                vec += line.strip().split()
                text += line.strip().split()
            else:
                data.append(vec)
                vec = []
            # if int((i / count) * 100) > pct:
            #     print(str(pct + 1) + '%')
    vocab = set(text)
    return data, vocab


def read_characters(file):
    chars = []
    vocab = set()
    count = 0
    # with open(file, encoding='ANSI') as f:
    with open(file, encoding='ascii', errors="surrogateescape") as f:
        while True:
            c = f.read(1)
            chars.append(c)
            vocab.add(c)
            count += 1
            if not c:
                print("End of file")
                break
            # print "Read a character:", c
    return chars, vocab


def check():
    chars, vocab = read_characters('dickens.txt')
    print(chars[:1000])
    counts = {i: chars.count(i) for i in set(chars)}
    print(counts)
    from math import log2
    # entropy
    def shannon(boe):
        total = sum(boe.values())
        return sum(freq / total * log2(total / freq) for freq in boe.values())
    print(shannon(counts))

# check()


# write2text function

# LSTM hoffman


class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return other.freq > self.freq


class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    # make probability dictionaries with sorted value from low to high !SPRINGER! put here the LSTM model
    def make_frequency_dict(self, context, model, corpus, device):
        model.eval()
        context_data = train.batchify(corpus.context_tokenize(context), 1, device)
        # print(context_data.size(0))
        data, targets = train.get_batch(280, context_data, 0)
        # print('data:', data)
        # print('targets:', targets)
        hidden = model.init_hidden(bsz=1)
        output, hidden = model(data, hidden)
        # model returns log softmax
        preds = output[-1].squeeze().exp().cpu().tolist()
        hidden = train.repackage_hidden(hidden)
        assert len(corpus.dictionary.idx2word) == len(preds)
        probs = {key: prob for key, prob in zip(corpus.dictionary.idx2word, preds)}
        # counted = dict(collections.Counter(text))
        sort = collections.OrderedDict(
            sorted(
                probs.items(),
                key=operator.itemgetter(1),
                reverse=False))
        threshold = 1e-3
        d = {}
        for key, prob in zip(corpus.dictionary.idx2word, preds):
            if prob > threshold:
                d[key] = prob
        print('d len', len(d))
        sort = collections.OrderedDict(
            sorted(
                d.items(),
                key=operator.itemgetter(1),
                reverse=False))

        # sort = sort[:5]
        # print(sort)

        # dummy = {'a': 0.5, 'b': 0.2, 'c': 0.2, 'd': 0.07, 'e': 0.03}
        # sort = collections.OrderedDict(
        #         sorted(
        #         dummy.items(),
        #         key=operator.itemgetter(1),
        #         reverse=False))

        return sort

    # make a heap queue from node
    def make_heap_node(self, freq_dict):
        for key in freq_dict:
            anode = HeapNode(key, freq_dict[key])
            self.heap.append(anode)

    # build tree
    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            # print('node1', node1.char, node1.freq)
            node2 = heapq.heappop(self.heap)
            # print('node2', node2.char, node2.freq)
            merge = HeapNode(None, node1.freq + node2.freq)
            merge.left = node1
            merge.right = node2
            heapq.heappush(self.heap, merge)

    # actual coding happens here
    def encode_helper(self, root, current_code):
        if root is None:
            return
        if root.char is not None:
            # try:
            #     print(root.char)
            # except UnicodeEncodeError:
            #     print('problem char')
            self.codes[root.char] = current_code
            return
        self.encode_helper(root.left, current_code + "0")
        self.encode_helper(root.right, current_code + "1")

    def encode(self):
        root = heapq.heappop(self.heap)
        current_code = ''
        self.encode_helper(root, current_code)

    def get_encoded_word(self, word):
        encoded_text = ''
        encoded_text += self.codes[word]
        return encoded_text


class HuffmanLSTMCompressor:
    def __init__(self, fname):
        self.fname = fname
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    """
    padding and eof
    https://www.cs.duke.edu/csed/poop/huff/info/#pseudo-eof
    https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1174/handouts/210%20Huffman%20Encoding.pdf
    """

    @staticmethod
    def pad_encoded_text(encoded_text):
        # get the extra padding of encoded text
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += '0'
        # merge the "info" of extra padding in "string/bit" with encoded text
        # so we know how to truncate it later
        padded_info = "{0:08b}".format(extra_padding)
        new_text = padded_info + encoded_text
        return new_text

    @staticmethod
    def to_byte_array(padded_encoded_text):
        if len(padded_encoded_text) % 8 != 0:
            print('not padded properly')
            exit(0)
        b = bytearray()
        for i in range(
                0, len(padded_encoded_text), 8):  # loop every 8 character
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))  # base 2
        return b

    def train_model(self):
        # train
        # return model
        pass

    def load_model(self, checkpoint='model.pt'):
        with open(checkpoint, 'rb') as f:
            model = torch.load(f).to(self.device)
        model.eval()
        return model

    def compress(self, filename):
        start = time.time()
        data, vocab = read_characters(self.fname)
        comp_text = ''
        corpus = Corpus(filename)

        # window size
        k = 10
        # actual window size (we predict last character based on previous k)
        k += 1
        # add start symbols. notice that we also need to add them on training time...
        data = ['<s>'] * (k - 1) + data
        # generator
        g = (data[t:t+k] for t in range(len(data) - k + 1))
        i = 0
        model = self.load_model()
        for window in g:
            i += 1
            # print(window)
            if i > 150:
                break
            # predict last char in window based on previous
            context = window[:-1]
            char = window[-1]
            huffman = HuffmanCoding()

            # !SPRINGER! put here the prediction of the model

            # if i == 147:
            # it was th... predict 'e' :)
            prob = huffman.make_frequency_dict(context, model, corpus, self.device)  # !SPRINGER! change it to the form you prefer.
            huffman.make_heap_node(prob)
            huffman.merge_nodes()
            huffman.encode()
            comp_text += huffman.get_encoded_word(char)

            # comp_text += huffman.get_encoded_word('e')
            # comp_text += huffman.get_encoded_word('a')
            # comp_text += huffman.get_encoded_word('b')
            # comp_text += huffman.get_encoded_word('d')
            # comp_text += huffman.get_encoded_word('c')

        print(comp_text)
        exit()

        padded_encoded_text = self.pad_encoded_text(comp_text)
        # is this really necessary? more bits... check remove_padding
        print(padded_encoded_text)
        byte_array_huff = self.to_byte_array(padded_encoded_text)

        # write header !SPRINGER! we will switch here to the information about the nn-LSTM.
        filename_split = filename.split('.')
        js = open(filename_split[0] + "_compressed.bin", 'wb')
        strcode = str(self.codes)
        js.write(strcode.encode())
        js.close()

        # append new line for separation
        append = open(filename_split[0] + "_compressed.bin", 'a')
        append.write('\n')
        append.close()

        # append the rest of the "byte array"
        f = open(filename_split[0] + "_compressed.bin", 'ab')
        f.write(bytes(byte_array_huff))
        f.close()

        # MISC
        print('Compression Done!')
        get_original_filesize = os.path.getsize(filename)
        get_compressed_filesize = os.path.getsize(
            filename_split[0] + "_compressed.bin")
        percentage = (get_compressed_filesize / get_original_filesize) * 100
        print(round(percentage, 3), "%")
        end = time.time()
        print(round((end - start), 3), "s")

    @staticmethod
    def remove_padding(padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)
        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-extra_padding]
        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ''
        decoded_text = ''
        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ''
        return decoded_text

    def decompress(self, compressedfile):  # !DROR! finished the decompress.
        start = time.time()
        filename_split = compressedfile.split('_')
        # get "header" !SPRINGER! we need to think how to encode/decode the information about LSTM. your call!
        header = open(compressedfile, 'rb').readline().decode()
        # header as object literal
        header = ast.literal_eval(header)
        # reverse mapping for better performance
        self.reverse_mapping = {v: k for k, v in header.items()}
        # get body
        f = open(compressedfile, 'rb')
        # get "body" as list.  [1:] because header
        body = f.readlines()[1:]
        f.close()
        bit_string = ""
        # merge list "body"
        # flattened the byte array
        join_body = [item for sub in body for item in sub]
        for i in join_body:
            bit_string += "{0:08b}".format(i)
        encoded_text = self.remove_padding(bit_string)
        # decompress start here
        current_code = ""
        decoded_text = ""
        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                decoded_text += self.reverse_mapping[current_code]
                current_code = ""
        write = open(filename_split[0] + "_decompressed.txt", 'w')
        write.writelines(decoded_text)
        write.close()
        print('Decompression Done!')
        end = time.time()
        print(round((end - start), 3), "s")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    # parser.add_argument('--data', type=str, default='./data',
    #                     help='location of the data corpus')
    # parser.add_argument('--model', type=str, default='LSTM',
    #                     help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    # parser.add_argument('--emsize', type=int, default=200,
    #                     help='size of word embeddings')
    # parser.add_argument('--nhid', type=int, default=200,
    #                     help='number of hidden units per layer')
    # parser.add_argument('--nlayers', type=int, default=2,
    #                     help='number of layers')
    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='initial learning rate')
    # parser.add_argument('--clip', type=float, default=0.25,
    #                     help='gradient clipping')
    # parser.add_argument('--epochs', type=int, default=1,
    #                     help='upper epoch limit')
    # parser.add_argument('--batch_size', type=int, default=20, metavar='N',
    #                     help='batch size')
    # parser.add_argument('--bptt', type=int, default=280,
    #                     help='sequence length')
    # parser.add_argument('--dropout', type=float, default=0.2,
    #                     help='dropout applied to layers (0 = no dropout)')
    # parser.add_argument('--tied', action='store_true',
    #                     help='tie the word embedding and softmax weights')
    # parser.add_argument('--seed', type=int, default=1111,
    #                     help='random seed')
    # parser.add_argument('--cuda', action='store_true',
    #                     help='use CUDA')
    # parser.add_argument('--log-interval', type=int, default=200, metavar='N',
    #                     help='report interval')
    # parser.add_argument('--save', type=str, default='model.pt',
    #                     help='path to save the final model')
    # parser.add_argument('--onnx-export', type=str, default='',
    #                     help='path to export the final model in onnx format')
    #
    # parser.add_argument('--nhead', type=int, default=2,
    #                     help='the number of heads in the encoder/decoder of the transformer model')
    #
    # args = parser.parse_args()

    huffmanLSTM = HuffmanLSTMCompressor(sys.argv[2])
    if sys.argv[1] == 'compress':
        huffmanLSTM.compress(sys.argv[2])
    elif sys.argv[1] == 'decompress':
        huffmanLSTM.decompress(sys.argv[2])
    else:
        print("command not found")
        exit(0)
    pass
