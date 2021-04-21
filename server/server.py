import sys
import json
import glob
import pickle
import torch
from flask import Flask, request

sys.path.append('../src')

from run import get_model, get_spm
from models.predictor_builder import build_predictor
from modules.data_loader import DataBatch
from utils.logger import init_logger, logger
from preprocess.lda.topic_model import TopicModel

app = Flask(__name__)
app.jinja_env.auto_reload = True

vocab_path = '../vocab/spm9998_3.model'
prodlda_vocab_file = '../models/prodlda/vocab.pkl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = '../models/tpt/model_step_200000.pt'
prodlda_checkpoint_path = '../models/prodlda_src/prodlda_model.pt'
model_name = 'TPT'
data_path = '../../data/MultiNewsTopicAll'


def load_dataset():

    def _dataset_loader(pt_file):
        file = json.load(open(pt_file))
        logger.info('Loading dataset from %s, number of examples: %d' %
                    (pt_file, len(file)))
        return file

    pts = sorted(glob.glob(data_path + '/test/*.[0-1].json'))  #
    dataset = []
    assert pts
    for pt in pts:
        dataset.extend(_dataset_loader(pt))

    return dataset


def get_prodlda_vocab(vocab_file):
    with open(vocab_file, 'rb') as file:
        vocab = pickle.load(file)
    return vocab


init_logger('./server.log')

spm, symbols = get_spm(vocab_path)
logger.info('Loading multi-document summarization model from %s' % checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
args = checkpoint['opt']
args.batch_size = 1

prodlda_vocab = get_prodlda_vocab(prodlda_vocab_file)
prodlda = TopicModel(prodlda_vocab, device, prodlda_checkpoint_path)

model = get_model(args, symbols, spm, device, checkpoint)
predictor = build_predictor(args, spm, symbols, model, device)
data = load_dataset()
print(len(data))


@app.route('/api/getData', methods=['GET'])
def get_data():
    index = int(request.args.get('id'))
    n_topic_words = int(request.args.get('nTopicWords'))
    example = data[index]

    srcs = [spm.DecodeIds(src) for src in example['src']]
    topk_scores, topk_indices, topk_words = prodlda.get_srcs_topic_words(srcs, n_topic_words)

    src_topic = [spm.Decode(topic) for topic in example['src_topic']]

    return {'target': example['tgt_str'], 'topicWords': topk_words, 'src': srcs, 'srcTopic': src_topic}


@app.route('/api/getSummary', methods=['POST'])
def predict():
    msg = json.loads(request.data)
    print(msg)
    topic_words, index = msg['topics'], int(msg['id'])

    ex = data[index]
    src, tgt, tgt_str, graph, para_topic = \
        ex['src'], ex['tgt'], ex['tgt_str'], ex['sim_graph'], ex['src_topic']

    src = src[:args.max_para_num]
    src = [para[:args.max_para_len] for para in src]

    graph = graph[:args.max_para_num]
    graph = [sim[:args.max_para_num] for sim in graph]

    tgt = tgt[:-1][:args.max_tgt_len] + [symbols['EOS']]
    tgt_ids = tgt[:-1]
    label_ids = tgt[1:]

    tgt_topic = [spm.Encode(word)[0] for word in topic_words]

    inst = [[src, tgt_ids, label_ids, tgt_str, graph, tgt_topic, para_topic]]

    batch = DataBatch(args.n_heads, args.max_para_num, args.max_para_len,
                      args.max_tgt_len, args.num_topic_words,
                      data=inst, pad_idx=symbols['PAD'], device=device, is_test=True)

    with torch.no_grad():
        results = predictor.translate_batch(batch)
    translation = predictor.from_batch(results)[0]
    pred, gold, src = translation
    pred_str = ' '.join(pred).replace('<Q>', ' ').replace(' +', ' ') \
        .replace('<unk>', 'UNK').replace('\\', '').strip()
    print(pred_str)
    return {'id': index, 'summary': pred_str, 'topicWords': topic_words}
