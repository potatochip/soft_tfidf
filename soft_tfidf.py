"""
Improved over the original in a few ways:
Terms not seen in the corpus are not ignored. Instead appropriate tfidf weight is given to them.
Close similarity terms use the partner term idf if it is available.
Additional option to calculate cos sim from the swapped vectors, but this tends to overweight
duplicate tokens at the moment.
"""
from collections import namedtuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from jellyfish import jaro_winkler


class SemiSoftTfidf(object):
    nan_term = u'_UNKNOWN_'
    similar = namedtuple('Similar', ['r1', 'r2', 'sim'])
    vectorizer_arguments = {
        'tokenizer': lambda x: x.split(),
        'smooth_idf': False,
        'sublinear_tf': True,
        'lowercase': False,
        'norm': None  # normalize later in case we need to weight on the fly
    }

    def __init__(self, corpus, verbose=True):
        self.corpus = corpus
        self.vectorizer = None
        self.vocabulary = None
        self.tfidf_dict = {}
        self._build_dict()

    def __getstate__(self):
        return {'tfidf_dict': self.tfidf_dict,
                'corpus': self.corpus,
                'vocabulary': self.vocabulary,
                'idf': self.vectorizer.idf_}

    def __setstate__(self, d):
        idf = d.pop('idf')
        self.vectorizer = self._inject_vectorizer(idf, d['vocabulary'])
        self.__dict__.update(d)

    def _inject_vectorizer(self, idf, vocabulary):
        TfidfVectorizer.idf_ = idf
        vectorizer = TfidfVectorizer(**self.vectorizer_arguments)
        vectorizer._tfidf._idf_diag = sp.spdiags(
            idf,
            diags=0,
            m=len(idf),
            n=len(idf)
        )
        vectorizer.vocabulary_ = vocabulary
        return vectorizer

    def _build_dict(self):
        corpus = [self._sorted_terms(i) for i in self.corpus]
        self.vectorizer = TfidfVectorizer(**self.vectorizer_arguments)
        matrix = self.vectorizer.fit_transform(corpus)
        self.vocabulary = self.vectorizer.vocabulary_

        for ix, doc in enumerate(corpus):
            weight_vector = [self._token_weight(matrix, ix, word) for word in doc.split()]
            self.tfidf_dict[doc] = self._normalize_list(weight_vector)

    def _token_weight(self, matrix, ix, word):
        return matrix[ix, self.vocabulary[word]]

    @staticmethod
    def _sorted_tokens(x):
        return sorted(x.split())

    @classmethod
    def _sorted_terms(cls, x):
        return u' '.join(cls._sorted_tokens(x))

    def similarity(self, x, y, threshold=0.95, return_cos_sim=False):
        self.x_bag = self._sorted_tokens(x)
        self.y_bag = self._sorted_tokens(y)
        x_bag = self.x_bag
        y_bag = self.y_bag

        x_alt, y_alt, sim_pairs = self._get_similar_pairs(x_bag, y_bag, threshold)
        new_x_bag = self._swap_unseen_tokens(x_bag, x_alt)
        new_y_bag = self._swap_unseen_tokens(y_bag, y_alt)

        x_idf = self._get_idf_vector(new_x_bag, x_bag)
        y_idf = self._get_idf_vector(new_y_bag, y_bag)

        if return_cos_sim:
            return self._cos_sim(x_bag, y_bag, x_idf, y_idf, y_alt)

        sim_pairs.sort(reverse=True, key=lambda x: x.sim)
        x_used = np.array([False] * len(x_bag), dtype=bool)
        y_used = np.array([False] * len(y_bag), dtype=bool)

        sim = 0.0
        for s in sim_pairs:
            if(x_used[s.r1] | y_used[s.r2]):
                continue
            x_bag_idf = x_idf[s.r1]
            y_bag_idf = y_idf[s.r2]
            sim += s.sim * x_bag_idf * y_bag_idf
            x_used[s.r1] = True
            y_used[s.r2] = True
        return float(sim)

    def _cos_sim(self, x_bag, y_bag, x_idf, y_idf, y_alt):
        # TODO: currently not symmetric
        tokens = set(x_bag)
        tokens.update(set(y_bag))
        token_column_d = {i: ix for ix, i in enumerate(tokens)}
        x = np.zeros(len(token_column_d))
        y = np.zeros(len(token_column_d))
        for ix, idf in enumerate(x_idf):
            token = x_bag[ix]
            tf = float(1 + np.log(x_bag.count(token))) / (len(x_bag) or 1)
            column = token_column_d[token]
            x[column] = tf * idf
        for ix, idf in enumerate(y_idf):
            token = y_bag[ix]
            tf = float(1 + np.log(y_bag.count(token))) / (len(y_bag) or 1)
            token = y_alt.get(token, token)
            column = token_column_d[token]
            y[column] = tf * idf
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        return cosine_similarity(x, y)[0][0]

    @staticmethod
    def _normalize_list(l):
        return normalize(np.array(l).reshape(1, -1), copy=False)[0]

    def _get_idf_vector(self, tokens, raw_tokens):
        term = ' '.join(raw_tokens)
        if term in self.tfidf_dict:
            return self.tfidf_dict[term]
        return self._transform_term(term, tokens, raw_tokens)

    def _transform_term(self, term, tokens, raw_tokens):
        doc = ' '.join(tokens)
        matrix = self.vectorizer.transform([doc])

        def calculate_weight(token):
            tf = float(1 + np.log(raw_tokens.count(token))) / (len(tokens) or 1)
            df = 0
            df += 1 if token in self.x_bag else 0
            df += 1 if token in self.y_bag else 0
            idf = np.log(float(len(self.corpus)) / df) + 1.0
            return tf * idf
        weight_vector = [
            calculate_weight(raw_tokens[ix]) if word == self.nan_term
            else self._token_weight(matrix, 0, word)
            for ix, word in enumerate(tokens)
        ]
        unit_norm_weight_vector = self._normalize_list(weight_vector)
        self.tfidf_dict[term] = unit_norm_weight_vector
        return unit_norm_weight_vector

    def _swap_unseen_tokens(self, tokens, sim_dict):
        new_tokens = []
        for ix, t in enumerate(tokens):
            alt = sim_dict.get(t)
            if t in self.vocabulary:
                new_tokens.append(t)
            elif alt in self.vocabulary:
                new_tokens.append(alt)
            else:
                new_tokens.append(self.nan_term)
        return new_tokens

    def _get_similar_pairs(self, x_bag, y_bag, threshold):
        sim_pairs = []
        x_alt = {}
        y_alt = {}
        for x_ix, s_token in enumerate(x_bag):
            for y_ix, t_token in enumerate(y_bag):
                dist = jaro_winkler(s_token, t_token)
                if dist >= threshold:
                    sim_pairs.append(self.similar(x_ix, y_ix, dist))
                    x_alt[x_bag[x_ix]] = y_bag[y_ix]
                    y_alt[y_bag[y_ix]] = x_bag[x_ix]

        return x_alt, y_alt, sim_pairs

    @staticmethod
    def evaluate_matrix_pair(row, inner_function):
        return [inner_function(x, y) for x, y in row]

    @staticmethod
    def evaluate_matrix_inner(row, inner_function):
        return [inner_function(x) for x in row]

    @staticmethod
    def apply_argsort(a, b, axis=-1):
        # if a == b then will return sorted a
        # if a != b then will return b sorted by a
        i = list(np.ogrid[[slice(x) for x in a.shape]])
        i[axis] = a.argsort(axis)
        return b[i]
