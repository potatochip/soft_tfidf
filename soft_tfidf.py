"""
Improved over the original in a few ways:
Terms not seen in the corpus are not ignored. Instead appropriate tfidf weight is given to them.
Close similarity terms use the partner term idf if it is available.
Additional option to calculate cos sim from the swapped vectors, but this tends to overweight
duplicate tokens at the moment.
"""
import bisect
import operator
from collections import Counter, namedtuple

import numpy as np
import scipy.sparse as sp
from jellyfish import jaro_winkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import normalize


class SoftTfidf(object):
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

    def similarity(self, x, y, threshold=0.95):
        self.x_bag = self._sorted_tokens(x)
        self.y_bag = self._sorted_tokens(y)
        x_bag = self.x_bag
        y_bag = self.y_bag

        x_alt, y_alt, sim_pairs = self._get_similar_pairs(x_bag, y_bag, threshold)
        new_x_bag = self._swap_unseen_tokens(x_bag, x_alt)
        new_y_bag = self._swap_unseen_tokens(y_bag, y_alt)

        x_idf = self._get_idf_vector(new_x_bag, x_bag)
        y_idf = self._get_idf_vector(new_y_bag, y_bag)

        sim_pairs.sort(reverse=True, key=lambda x: x.sim)
        x_used = np.array([False] * len(x_bag), dtype=bool)
        y_used = np.array([False] * len(y_bag), dtype=bool)

        sim = 0.0
        for s in sim_pairs:
            if x_used[s.r1] | y_used[s.r2]:
                continue
            x_bag_idf = x_idf[s.r1]
            y_bag_idf = y_idf[s.r2]
            sim += s.sim * x_bag_idf * y_bag_idf
            x_used[s.r1] = True
            y_used[s.r2] = True
        return float(sim)

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


class SemiSoftTfidf(object):
    # BUG: unknown terms are currently counted as multiple same terms
    nan_term = u'_UNKNOWN_'

    def __init__(self, corpus, threshold=0.9):
        self.corpus = corpus
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(norm=None, sublinear_tf=True, smooth_idf=True, lowercase=False, tokenizer=lambda x: x.split())
        self.matrix = normalize(self.vectorizer.fit_transform(corpus))
        self.column_lookup = self.vectorizer.vocabulary_
        self.sorted_terms = sorted(i for i in self.column_lookup.keys())
        self.probability_lookup = self._probabilities()
        self._max_idx = len(self.sorted_terms)
        self._min_idx = 0

    def _probabilities(self):
        count = Counter()
        for doc in self.corpus:
            for token in set(doc.split()):
                count[token] += 1
        max_count = len(count)
        return {k: float(v)/max_count for k, v in count.items()}

    def _closest_lexicographic_idx(self, term):
        return bisect.bisect_left(self.sorted_terms, term)

    def _windowed_distance(self, term, window=1000):
        idx = self._closest_lexicographic_idx(term)
        windowed_terms = self.sorted_terms[max(idx-window, self._min_idx): min(idx+window, self._max_idx)]
        return sorted([(jaro_winkler(t, term), t) for t in windowed_terms], reverse=True)

    def _break_tie(self, terms):
        return max((self.probability_lookup[t], t) for t in terms)[1]

    def _max_distances(self, term):
        max_list = []
        current_max = self.threshold
        for i in self._windowed_distance(term):
            if i[0] >= current_max:
                max_list.append(i)
                current_max = i[0]
            else:
                break
        return max_list

    def _best_match(self, term):
        if term in self.sorted_terms:
            return (1.0, term)
        max_list = self._max_distances(term)
        num_maxes = len(max_list)
        if num_maxes == 0:
            return (0.0, term)
        elif num_maxes == 1:
            return max_list[0]
        else:
            tie_broken_term = self._break_tie(i[1] for i in max_list)
            return (max_list[0][0], tie_broken_term)

    def retrieve(self, query, best_matches=False):
        query = unicode(query)
        t_bag = query.split()
        sims, terms = zip(*[self._best_match(t) for t in t_bag])
        transformed = self.vectorizer.transform([' '.join(terms)])
        dist_matrix = sp.csr_matrix((sims, ([0]*len(terms), [self.column_lookup[t] for t in terms])), transformed.shape)
        comparison_vector = normalize(transformed.multiply(dist_matrix))
        cos_sim = linear_kernel(self.matrix, comparison_vector).ravel()
        if best_matches:
            partial_sort_indexes = np.argpartition(cos_sim, np.arange(-5, 0, 1), axis=0)
            closest_idx = reversed(partial_sort_indexes[-5:])
            return operator.itemgetter(*closest_idx)(self.corpus)
        closest_idx = np.argmax(cos_sim)
        return self.corpus[closest_idx]
