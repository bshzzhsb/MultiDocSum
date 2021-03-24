import gensim
import argparse

from preprocess.utils import data_loader

parser = argparse.ArgumentParser()

# data and file related arguments
parser.add_argument('--data_path', type=str, default='../../data/MultiNews')
parser.add_argument('--emb_file', type=str, default='../models/embeddings.txt', help='file to save the word embeddings')
parser.add_argument('--dim_rho', type=int, default=300, help='dimensionality of the word embeddings')
parser.add_argument('--min_count', type=int, default=10, help='minimum term frequency (to define the vocabulary)')
parser.add_argument('--sg', type=int, default=1, help='whether to use skip-gram')
parser.add_argument('--workers', type=int, default=25, help='number of CPU cores')
parser.add_argument('--negative_samples', type=int, default=10, help='number of negative samples')
parser.add_argument('--window_size', type=int, default=4, help='window size to determine context')
parser.add_argument('--iters', type=int, default=50, help='number of iterations')

args = parser.parse_args()

# Class for a memory-friendly iterator over the dataset
dataset = data_loader(args.data_path)

# Gensim code to obtain the embeddings
model = gensim.models.Word2Vec(dataset, min_count=args.min_count, sg=args.sg, size=args.dim_rho,
                               iter=args.iters, workers=args.workers, negative=args.negative_samples,
                               window=args.window_size)
model.save('../models/w2v_300.model')

# Write the embeddings to a file
with open(args.emb_file, 'w') as f:
    for v in list(model.wv.vocab):
        vec = list(model.wv.__getitem__(v))
        f.write(v + ' ')
        vec_str = ['%.9f' % val for val in vec]
        vec_str = " ".join(vec_str)
        f.write(vec_str + '\n')
