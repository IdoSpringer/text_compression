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
from data import Corpus, Context, Dictionary
import torch
from model import RNNModel
import argparse
import torch.nn.functional as F
import io

# todo threshold in huffman dictionary before building the tree
# todo train and compress end-to-end


def read_characters(file):
    chars = []
    vocab = set()
    count = 0
    with open(file, encoding='ascii', errors="surrogateescape") as f:
        while True:
            c = f.read(1)
            chars.append(c)
            vocab.add(c)
            count += 1
            if not c:
                break
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
        self.vocab = {}

    @staticmethod
    def make_frequency_dict(context, model, context_map, device):
        model.eval()
        context_data = train.batchify(context_map.context_tokenize(context), 1, device)
        data, targets = train.get_batch(280, context_data, 0)
        with torch.no_grad():
            hidden = model.init_hidden(bsz=1)
            output, hidden = model(data, hidden)
            # model returns log softmax
            preds = output[-1].squeeze().exp().cpu().tolist()
            hidden = train.repackage_hidden(hidden)
        assert len(context_map.dictionary.idx2word) == len(preds)
        probs = {key: prob for key, prob in zip(context_map.dictionary.idx2word, preds)}
        sort = collections.OrderedDict(
            sorted(
                probs.items(),
                key=operator.itemgetter(1),
                reverse=False))

        # try threshold - later
        # threshold = 1e-3
        # threshold = 0
        # d = {}
        # for key, prob in zip(context_map.dictionary.idx2word, preds):
        #     if prob > threshold:
        #         d[key] = prob
        # # print('d len', len(d))
        # sort = collections.OrderedDict(
        #     sorted(
        #         d.items(),
        #         key=operator.itemgetter(1),
        #         reverse=False))
        # for key in sort:
        #     print(key, sort[key])
        # exit()
        # sort = sort[:5]
        # print(sort)

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
            node2 = heapq.heappop(self.heap)
            merge = HeapNode(None, node1.freq + node2.freq)
            merge.left = node1
            merge.right = node2
            heapq.heappush(self.heap, merge)

    # actual coding happens here
    def encode_helper(self, root, current_code):
        if root is None:
            return
        if root.char is not None:
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
        # actually it is reasonable to train one epoch in compress time, make it end-to-end
        with open(checkpoint, 'rb') as f:
            model = torch.load(f).to(self.device)
        model.eval()
        return model

    def compress(self, filename):
        start = time.time()
        data, vocab = read_characters(self.fname)
        comp_text = ''
        dictionary = Corpus(filename).dictionary
        context_map = Context(dictionary)
        # window size
        k = 10
        # actual window size (we predict last character based on previous k)
        k += 1
        # start symbols
        data = ['<s>'] * (k - 1) + data
        # generator
        g = (data[t:t+k] for t in range(len(data) - k + 1))
        i = 0
        model = self.load_model()
        for window in g:
            i += 1
            print(i)
            # first two paragraphs
            if i > 929:
                break
            # predict last char in window based on previous 10
            context = window[:-1]
            char = window[-1]
            huffman = HuffmanCoding()
            prob = huffman.make_frequency_dict(context, model, context_map, self.device)
            huffman.make_heap_node(prob)
            huffman.merge_nodes()
            huffman.encode()
            comp_text += huffman.get_encoded_word(char)
        padded_encoded_text = self.pad_encoded_text(comp_text)
        byte_array_huff = self.to_byte_array(padded_encoded_text)
        filename_split = filename.split('.')
        compressed_filename = filename_split[0] + "_compressed.bin"
        # write compressed file
        torch.save({
            'word2idx': dictionary.word2idx,
            'idx2word': dictionary.idx2word,
            'model_state_dict': model.state_dict(),
            'bytes': bytes(byte_array_huff),
        }, compressed_filename)
        # size in bytes
        print(os.path.getsize(compressed_filename))
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

    def decompress(self, compressedfile):
        start = time.time()
        filename_split = compressedfile.split('_')
        checkpoint = torch.load(compressedfile, map_location=self.device)
        body = checkpoint['bytes']
        dictionary = Dictionary()
        dictionary.word2idx = checkpoint['word2idx']
        dictionary.idx2word = checkpoint['idx2word']
        context_map = Context(dictionary)
        ntokens = len(dictionary)
        model = RNNModel('LSTM', ntokens, 200, 200, 2, dropout=0.2, tie_weights=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        bit_string = ''
        join_body = list(body)
        for i in join_body:
            bit_string += "{0:08b}".format(i)
        encoded_text = self.remove_padding(bit_string)
        # decompress start here
        current_code = ''
        decoded_text = ''
        # we define an initial context
        # then we predict the initial huffman tree
        # read bits until we get to a leaf
        # convert the leaf to a char and add it to decompressed text
        # update the context and repeat the process
        context = ['<s>'] * 10
        def tree_from_context(context):
            huffman = HuffmanCoding()
            prob = huffman.make_frequency_dict(context, model, context_map, self.device)
            huffman.make_heap_node(prob)
            huffman.merge_nodes()
            huffman.encode()
            huffman.reverse_mapping = {v: k for k, v in huffman.codes.items()}
            return huffman
        huffman = tree_from_context(context)
        for bit in encoded_text:
            current_code += bit
            if current_code in huffman.reverse_mapping:
                next_char = huffman.reverse_mapping[current_code]
                decoded_text += next_char
                current_code = ''
                context = context[1:] + [next_char]
                huffman = tree_from_context(context)
        # write decompressed file
        with open(filename_split[0] + "_decompressed.txt", 'w') as f:
            f.writelines(decoded_text)
        print('Decompression Done!')
        end = time.time()
        print(round((end - start), 3), "s")


if __name__ == '__main__':
    huffmanLSTM = HuffmanLSTMCompressor(sys.argv[2])
    if sys.argv[1] == 'compress':
        huffmanLSTM.compress(sys.argv[2])
    elif sys.argv[1] == 'decompress':
        huffmanLSTM.decompress(sys.argv[2])
    else:
        print("command not found")
        exit(0)
    pass
