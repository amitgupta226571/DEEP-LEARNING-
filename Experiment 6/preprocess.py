import unicodedata
import re
import os
import numpy as np

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()
    return '<start> ' + w + ' <end>'

def load_dataset(path, num_examples=None):
    if not os.path.exists(path):
        return None, None
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')[:2]] for l in lines[:num_examples]]
    return zip(*word_pairs)

class Tokenizer:
    def __init__(self):
        self.word2idx = {"<pad>": 0}
        self.idx2word = {0: "<pad>"}
        self.vocab_size = 1

    def fit_on_texts(self, texts):
        for sentence in texts:
            for word in sentence.split(' '):
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def texts_to_sequences(self, texts):
        return [[self.word2idx[word] for word in sentence.split(' ')] for sentence in texts]

def pad_sequences(sequences, padding='post'):
    max_len = max(len(seq) for seq in sequences)
    padded_seqs = []
    for seq in sequences:
        if padding == 'post':
            padded_seqs.append(seq + [0] * (max_len - len(seq)))
        else:
            padded_seqs.append([0] * (max_len - len(seq)) + seq)
    return np.array(padded_seqs)

def tokenize(lang):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lang)
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = pad_sequences(tensor, padding='post')
    return tensor, tokenizer
