"""
Given seed labels, entities and caption to rank candidate labels.

author: Shuo Zhang
"""

from column_evaluation import Column_evaluation
from scorer import ScorerLM
from elastic import Elastic
import math


class Rank_label(Column_evaluation):
    def __init__(self, index_name = "table_index_frt"):
        super().__init__()
        self.__tes = Elastic(index_name=index_name)
        self.__num = 100

    def rank_candidates(self, seed_label, E, c):
        """Ranking candidate labels"""
        p_all = {}
        labels_c, tables_c = self.find_candidates_c(c, seed=seed_label, num=self.__num)  # search tables with similar caption
        labels_e, tables_e = self.find_candidates_e(E, seed=seed_label, num=self.__num)
        lables_h, tables_h = self.find_candidates_l(seed=seed_label, num=self.__num)
        all_tables = set(tables_c + tables_e + tables_h) # all related tables (ids)
        candidate_labels = set(list(labels_c) + list(labels_e) + list(lables_h))
        p_t_ecl, headings = self.p_t_ecl(all_tables, seed_label, E)
        for label in candidate_labels:
            p_all[label] = 0
            for table in all_tables:
                table_label = headings.get(table,[])
                if label in table_label:
                    p_all[label] += p_t_ecl[table]/len(table_label)
        return p_all

    def p_t_ecl(self, all_table, seed_label, E):
        p = {}
        headings = {}
        for table in all_table:
            doc = self.__tes.get_doc(table)
            table_label = doc.get("_source").get("headings_n")
            headings[table] = table_label
            sim_l = self.overlap(table_label, seed_label)
            table_entity = doc.get("_source").get("entity")
            sim_e = self.overlap(table_entity, E)
            table_caption = doc.get("_source").get("caption")
            score = ScorerLM(self.__tes, table_caption, {}).score_doc(table)
            p[table] = max(sim_e, 0.000001) * max(sim_l, 0.000001) * max(math.exp(score), 0.000001)
        return p, headings


    def overlap(self, a, b):
        """Calculate |A and B|/|B|"""
        return len([i for i in a if i in b]) / len(b)


if __name__ == "__main__":
    r = Rank_label()
    seed_label = ["episode"]
    E = ["Does_the_Team_Think?"]
    c = "Episodes"
    res = r.rank_candidates(seed_label=seed_label, E=E, c=c)
    print(res)


