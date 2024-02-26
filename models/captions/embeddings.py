import torch
import numpy as np
from . import config


def load_glove_embeddings(embedding_file):
    word_vectors = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            word_vectors[word] = vector
    return word_vectors


def pretrained_weights(vocab, embedding_dim):
    pretrained_embeddings = load_glove_embeddings(config.glove_file)
    vocab_size = len(vocab)
    pretrained_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in vocab.get_stoi().items():
        if word in pretrained_embeddings:
            # Truncate or pad the embedding to match the desired dimension
            pretrained_matrix[idx][:len(pretrained_embeddings[word])] = pretrained_embeddings[word][:embedding_dim]

    return torch.FloatTensor(pretrained_matrix)
