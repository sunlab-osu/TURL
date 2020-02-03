"""
Evaluation of Column Population
-------------------------------

Column population pipeline.

@author: Shuo Zhang
"""

from elastic import Elastic



class Column_evaluation(object):
    def __init__(self, test_tables=None):
        """

        :param test_tables: 1000 wiki test tables
        """
        self.test_tables = test_tables
        self.__tes = Elastic("table_index_frt")

    def rank_candidates(self, seed, c, E):
        """

        :param cand: candidate entities
        :param seed: Seed entity
        :param c: Table caption
        :return: Ranked suggestions
        """
        pass

    def find_candidates_c(self, c, seed, num=100):
        """find candidate tables complement with table caption"""
        res = self.__tes.search(query=c, field="caption", num=num)
        cand = []
        for table_id in res.keys():
            doc = self.__tes.get_doc(table_id)
            labels = doc["_source"]["headings_a"]
            cand += labels
        return set([i for i in cand if i not in seed]), list(res.keys())


    def find_candidates_l(self, seed, num=100):
        """find candidate labels Using labels """
        tables = []
        cand = []
        for label in seed:
            res = self.__tes.search(query=label, field="headings", num=num)
            tables += list(res.keys())
            for table_id in res.keys():
                doc = self.__tes.get_doc(table_id)
                labels = doc["_source"]["headings"]
                cand += labels
        return set([i for i in cand if i not in seed]), tables


    def find_candidates_e(self, E, seed, num=100):
        """find candidate labels Using entities"""
        tables = []
        cand = []
        for entity in E:
            body = self.generate_search_body(entity=entity, field="entity")
            res = self.__tes.search_complex(body=body, num=num)
            tables += list(res.keys())
            for table_id in res.keys():
                doc = self.__tes.get_doc(table_id)
                labels = doc["_source"]["headings"]
                cand += labels
        return set([i for i in cand if i not in seed]), tables

    def generate_search_body(self, entity, field):
        """Generate search body"""
        body = {
            "query": {
                "bool": {
                    "must": {
                        "term": {field: entity}
                    }
                }
            }
        }
        return body

    def parse(self, text):
        """Put query into a term list for term iteration"""
        stopwords = []
        terms = []
        # Replace specific characters with space
        chars = ["'", ".", ":", ",", "/", "(", ")", "-", "+"]
        for ch in chars:
            if ch in text:
                text = text.replace(ch, " ")
        # Tokenization
        for term in text.split():  # default behavior of the split is to split on one or more whitespaces
            # Stopword removal
            if term in stopwords:
                continue
            terms.append(term)
        return terms





