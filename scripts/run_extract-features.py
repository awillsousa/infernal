# -*- coding: utf-8 -*-

"""
Extract features to be later used by shallow classifiers.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import numpy as np

from infernal import utils
from infernal import feature_extraction as fe
from infernal import openwordnetpt as own
from infernal import config

if __name__ == '__main__':
    pairs = utils.load_pickled_pairs("data/assin-ptbr-train.pickle")
    stopwords = utils.load_stopwords()
    own.load_wordnet(config.ownpt_path)

    vocab_path = utils.get_vocabulary_path("data/vetores.npy")
    ed = utils.EmbeddingDictionary(vocab_path, "data/vetores.npy")
    fex = fe.FeatureExtractor(True, stopwords, ed)

    feature_names = fex.get_feature_names()
    x = fex.extract_dataset_features(pairs)
    ld = utils.load_label_dict(None) if args.load_ld_path else None
    #y, ld = utils.extract_classes(pairs, ld)

    np.savez(args.output, x=x, y=y)
    if args.save_ld_path:
        utils.write_label_dict(ld, "data/dicionario.dict")
