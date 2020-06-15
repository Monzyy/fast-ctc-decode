from fast_rnn_lm_ctc_decode import *
from bonito.decode import load_rnn_lm
import numpy as np
from bonito.lm import RNNLanguageModel, load_rnn_lm

alphabet = "NACGT"

def get_random_data(samples=100):
    x = np.random.rand(samples, len(alphabet)).astype(np.float32)
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)

def main():
    beam_size = 5
    beam_cut_threshold = 0.1
    probs = get_random_data()
    alpha = 1.1
    beta = 0.9

    net = load_rnn_lm('/home/mac/workspaces/andreasbonito/bonito/lm.net', 'cuda')
    net = RNNLanguageModel(net)
    seq, path = beam_search(probs, alphabet, beam_size, beam_cut_threshold, net, alpha, beta)
    print(seq)


if __name__ == "__main__":
    main()