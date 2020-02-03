"""
Scorer
======

Various retrieval models for scoring a individual document for a given query.

:Authors: Faegheh Hasibi, Krisztian Balog
"""
import math
import sys

from elastic import Elastic
from elastic_cache import ElasticCache


class Scorer(object):
    """Base scorer class."""

    SCORER_DEBUG = 0

    def __init__(self, elastic, query, params):
        self._elastic = elastic
        self._query = query
        self._params = params

        # The analyser might return terms that are not in the collection.
        # These terms are filtered out later in the score_doc functions.
        if self._query:
            self._query_terms = elastic.analyze_query(self._query).split()
        else:
            self._query_terms = []

    # def score_doc(self, doc_id):
    #     """Scorer method to be implemented in each subclass."""
    #     # should use elastic scoring
    #     query = self._elastic.analyze_query(self._query)
    #     field = params["first_pass"]["field"]
    #     res = self._elastic.search(query, field, num=self.__first_pass_num_docs, start=start)
    #     return

    @staticmethod
    def get_scorer(elastic, query, config):
        """Returns Scorer object (Scorer factory).

        :param elastic: Elastic object
        :param query: raw query (to be analyzed)
        :param config: dict with models parameters
        """
        model = config.get("model", "lm")
        if model == "lm":
            return ScorerLM(elastic, query, config)
        elif model is None:
            return None
        else:
            raise Exception("Unknown model " + model)


# =========================================
# ================== LM  ==================
# =========================================
class ScorerLM(Scorer):
    """Language Model (LM) scorer."""
    JM = "jm"
    DIRICHLET = "dirichlet"

    def __init__(self, elastic, query, params):
        super(ScorerLM, self).__init__(elastic, query, params)
        self._field = params.get("fields", "catchall")
        self._smoothing_method = params.get("smoothing_method", self.DIRICHLET).lower()
        if self._smoothing_method == self.DIRICHLET:
            self._smoothing_param = params.get("smoothing_param", 50)
        elif self._smoothing_method == ScorerLM.JM:
            self._smoothing_param = params.get("smoothing_param", 0.1)
        # self._smoothing_param = params.get("smoothing_param", None)
        else:
            sys.exit(0)

        self._tf = {}

    @staticmethod
    def get_jm_prob(tf_t_d, len_d, tf_t_C, len_C, lambd):
        """Computes JM-smoothed probability.
        p(t|theta_d) = [(1-lambda) tf(t, d)/|d|] + [lambda tf(t, C)/|C|]

        :param tf_t_d: tf(t,d)
        :param len_d: |d|
        :param tf_t_C: tf(t,C)
        :param len_C: |C| = \sum_{d \in C} |d|
        :param lambd: \lambda
        :return: JM-smoothed probability
        """
        p_t_d = tf_t_d / len_d if len_d > 0 else 0
        p_t_C = tf_t_C / len_C if len_C > 0 else 0
        if Scorer.SCORER_DEBUG:
            print("\t\t\tp(t|d) = {}\tp(t|C) = {}".format(p_t_d, p_t_C))
        return (1 - lambd) * p_t_d + lambd * p_t_C

    @staticmethod
    def get_dirichlet_prob(tf_t_d, len_d, tf_t_C, len_C, mu):
        """Computes Dirichlet-smoothed probability.
        P(t|theta_d) = [tf(t, d) + mu P(t|C)] / [|d| + mu]

        :param tf_t_d: tf(t,d)
        :param len_d: |d|
        :param tf_t_C: tf(t,C)
        :param len_C: |C| = \sum_{d \in C} |d|
        :param mu: \mu
        :return: Dirichlet-smoothed probability
        """
        if mu == 0:  # i.e. field does not have any content in the collection
            return 0
        else:
            p_t_C = tf_t_C / len_C if len_C > 0 else 0
            return (tf_t_d + mu * p_t_C) / (len_d + mu)

    def __get_term_freq(self, doc_id, field, term):
        """Returns the (cached) term frequency."""
        if doc_id not in self._tf:
            self._tf[doc_id] = {}
        if field not in self._tf[doc_id]:
            self._tf[doc_id][field] = self._elastic.term_freqs(doc_id, field)
        return self._tf[doc_id][field].get(term, 0)

    def get_lm_term_prob(self, doc_id, field, t, tf_t_d_f=None, tf_t_C_f=None):
        """Returns term probability for a document and field.

        :param doc_id: document ID
        :param field: field name
        :param t: term
        :return: P(t|d_f)
        """
        len_d_f = self._elastic.doc_length(doc_id, field)
        len_C_f = self._elastic.coll_length(field)
        tf_t_C_f = self._elastic.coll_term_freq(t, field) if tf_t_C_f is None else tf_t_C_f
        tf_t_d_f = self.__get_term_freq(doc_id, field, t) if tf_t_d_f is None else tf_t_d_f
        if self.SCORER_DEBUG:
            print("\t\tt = {}\t f = {}".format(t, field))
            print("\t\t\tDoc:  tf(t,f) = {}\t|f| = {}".format(tf_t_d_f, len_d_f))
            print("\t\t\tColl: tf(t,f) = {}\t|f| = ".format(tf_t_C_f, len_C_f))

        p_t_d_f = 0
        # JM smoothing: p(t|theta_d_f) = [(1-lambda) tf(t, d_f)/|d_f|] + [lambda tf(t, C_f)/|C_f|]
        if self._smoothing_method == self.JM:
            lambd = self._smoothing_param
            p_t_d_f = self.get_jm_prob(tf_t_d_f, len_d_f, tf_t_C_f, len_C_f, lambd)
            if self.SCORER_DEBUG:
                print("\t\t\tJM smoothing:")
                print("\t\t\tDoc:  p(t|theta_d_f)= ", p_t_d_f)

        # Dirichlet smoothing
        elif self._smoothing_method == self.DIRICHLET:
            mu = self._smoothing_param if self._smoothing_param != "avg_len" else self._elastic.avg_len(field)
            p_t_d_f = self.get_dirichlet_prob(tf_t_d_f, len_d_f, tf_t_C_f, len_C_f, mu)
            if self.SCORER_DEBUG:
                print("\t\t\tDirichlet smoothing:")
                print("\t\t\tmu: ", mu)
                print("\t\t\tDoc:  p(t|theta_d_f)= ", p_t_d_f)
        return p_t_d_f

    def get_lm_term_probs(self, doc_id, field):
        """Returns probability of all query terms for a document and field; i.e. p(t|theta_d)

        :param doc_id: document ID
        :param field: field name
        :return: dictionary of terms with their probabilities
        """
        p_t_theta_d_f = {}
        for t in set(self._query_terms):
            p_t_theta_d_f[t] = self.get_lm_term_prob(doc_id, field, t)
        return p_t_theta_d_f

    def score_doc(self, doc_id):
        """Scores the given document using LM.
        p(q|theta_d) = \sum log(p(t|theta_d))

        :param doc_id: document id
        :return: LM score
        """
        if self.SCORER_DEBUG:
            print("Scoring doc ID=" + doc_id)

        p_t_theta_d = self.get_lm_term_probs(doc_id, self._field)
        if sum(p_t_theta_d.values()) == 0:  # none of query terms are in the field collection
            if self.SCORER_DEBUG:
                print("\t\tP(q|{}) = None".format(self._field))
            return None

        # p(q|theta_d) = sum log(p(t|theta_d)); we return log-scale values
        p_q_theta_d = 0
        for t in self._query_terms:
            # Skips the term if it is not in the field collection
            if p_t_theta_d[t] == 0:
                continue
            if self.SCORER_DEBUG:
                print("\t\tP({}|{}) = {}".format(t, self._field, p_t_theta_d[t]))
            p_q_theta_d += math.log(p_t_theta_d[t])
        if self.SCORER_DEBUG:
            print("P(d|q) = {}".format(p_q_theta_d))
        return p_q_theta_d


if __name__ == "__main__":
    query = "gonna friends"
    doc_id = "4"
    es = ElasticCache("toy_index")
    params = {"fields": "content",
              "__fields": {"title": 0.2, "content": 0.8},
              "__fields": ["content", "title"]
              }
    score = ScorerPRMS(es, query, params).score_doc(doc_id)
    print(score)
