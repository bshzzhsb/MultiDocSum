import os
import time
from pyrouge import Rouge155
from multiprocessing import Pool

from utils.logger import logger


def process(data):
    candidates, references, pool_id = data
    count = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = '../results/rouge-tmp-{}-{}'.format(current_time, pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + '/candidate')
        os.mkdir(tmp_dir + '/reference')
    for i in range(count):
        if len(references[i]) < 1:
            continue
        with open(tmp_dir + '/candidate/candi.{}.txt'.format(i), 'w', encoding='utf-8') as f:
            f.write(candidates[i])
        with open(tmp_dir + '/reference/ref.{}.txt'.format(i), 'w', encoding='utf-8') as f:
            f.write(references[i])

    r = Rouge155()
    r.model_dir = tmp_dir + '/reference/'
    r.system_dir = tmp_dir + '/candidate/'
    r.model_filename_pattern = 'ref.#ID#.txt'
    r.system_filename_pattern = r'candi.(\d+).txt'
    rouge_results = r.convert_and_evaluate()
    logger.info(rouge_results)
    results_dict = r.output_to_dict(rouge_results)

    return results_dict


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i: i + n]


def test_rouge(candi, ref, num_processes):
    candidates = [line.strip() for line in candi]
    references = [line.strip() for line in ref]

    assert len(candidates) == len(references)

    candidates_chunks = list(chunks(candidates, int(len(candidates) / num_processes)))
    references_chunks = list(chunks(references, int(len(references) / num_processes)))
    n_pool = len(candidates_chunks)
    arg_list = []
    for i in range(n_pool):
        arg_list.append((candidates_chunks[i], references_chunks[i], i))
    pool = Pool(n_pool)

    results = pool.map(process, arg_list)
    final_results = {}

    for i, r in enumerate(results):
        for k in r:
            if k not in final_results:
                final_results[k] = r[k] * len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])

    for k in final_results:
        final_results[k] = final_results[k] / len(candidates)

    return final_results


def rouge_results_to_str(results_dict):
    return ">> ROUGE_F(1/2/l): {:.2f}/{:.2f}/{:.2f}\n" \
           "ROUGE-R(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
                results_dict['rouge_1_f_score'] * 100,
                results_dict['rouge_2_f_score'] * 100,
                results_dict['rouge_l_f_score'] * 100,
                results_dict['rouge_1_recall'] * 100,
                results_dict['rouge_2_recall'] * 100,
                results_dict['rouge_l_recall'] * 100,
            )
