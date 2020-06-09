import numpy as np
import model
import os
import heapq
import collections
import operator
import ast
import sys
import time


# read function !DROR! to include '.' and others as words.
# def read_data(fname):
#     data = []
#     vec = []
#     text = []
#     count = len(open(fname).readlines())
#     idxl = 0
#     # with open(fname, 'rb') as f:
#     #     file = f.read()
#     with open(fname, encoding="utf-8", errors='ignore') as f:
#         for line in file:
#             idxl = idxl+1
#             if line.strip().split():
#                 vec = vec + line.strip().split()
#                 text = text + line.strip().split()
#             else:
#                 data.append(vec)
#                 vec = []
#             print(100*(idxl/count))
#     vocab = set(text)
#     return data, vocab

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


data, vocab = read_data('dickens.txt')
print(data[:100])
print(len(vocab))

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
    def make_frequency_dict(self, text):
        counted = dict(collections.Counter(text))
        sort = collections.OrderedDict(
            sorted(
                counted.items(),
                key=operator.itemgetter(1),
                reverse=False))
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
        current_code = ""
        self.encode_helper(root, current_code)

    def get_encoded_word(self, word):
        encoded_text = ""
        encoded_text += self.codes[word]
        return encoded_text


class HuffmanLSTM:
    def __init__(self, fname):
        self.fname = fname

    """
    padding and eof
    https://www.cs.duke.edu/csed/poop/huff/info/#pseudo-eof
    https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1174/handouts/210%20Huffman%20Encoding.pdf
    """

    def pad_encoded_text(self, encoded_text):
        # get the extra padding of encoded text
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"
        # merge the "info" of extra padding in "string/bit" with encoded text
        # so we know how to truncate it later
        padded_info = "{0:08b}".format(extra_padding)
        new_text = padded_info + encoded_text

        return new_text

    def to_byte_array(self, padded_encoded_text):
        if len(padded_encoded_text) % 8 != 0:
            print('not padded properly')
            exit(0)
        b = bytearray()
        for i in range(
                0, len(padded_encoded_text), 8):  # loop every 8 character
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))  # base 2
        return b


    def compress(self, filename):
        start = time.time()
        Data, Vocab = read_data(self.fname)
        # !SPRINGER! put here the training of the model

        #
        compText = ''
        for line in Data:
            for word in line:
                huffman = HuffmanCoding()
                prob = huffman.make_frequency_dict(word)  # !SPRINGER! change it to the form you prefer.
                huffman.make_heap_node(prob)
                huffman.merge_nodes()
                huffman.encode()
                compText += huffman.get_encoded_word(word)

        padded_encoded_text = self.pad_encoded_text(compText)
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

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)
        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-extra_padding]
        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""

        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""

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
    # huffmanLSTM = HuffmanLSTM(sys.argv[2])
    # if sys.argv[1] == 'compress':
    #     huffman.compress(sys.argv[2])
    # elif sys.argv[1] == 'decompress':
    #     huffman.decompress(sys.argv[2])
    # else:
    #     print("command not found")
    #     exit(0)
    pass