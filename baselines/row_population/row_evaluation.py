"""
Evaluation of Row Population
----------------------------

Row population performance evaluation.

@author: Shuo Zhang
"""

from elastic import Elastic
import pdb
import elasticsearch


class Row_evaluation(object):
    def __init__(self, type_index_name="dbpedia_2015_10_type_cat", table_index_name="table_index_frt"):
        """

        :param index_name: name of index
        """
        self.__index_name = type_index_name
        self.__elastic = Elastic(self.__index_name)
        self.__tes = Elastic(table_index_name)

    def rank_candidates(self, seed , c=None, l=None):
        """

        :param cand: candidate entities
        :param seed: Seed entity
        :param a: Attributes
        :param c: Table caption
        :return: Ranked suggestions
        """
        pass
    
    def find_candidates_c(self, seed_E, c, num=100):
        """table caption to find candidate entities"""
        cand = []
        res = self.__tes.search(query=c, field="catchall", num=num)
        for table_id in res.keys():
            doc = self.__tes.get_doc(table_id)
            labels = doc["_source"]["entity_n"]
            cand += labels
        return set([i for i in cand if i not in seed_E])

    def find_core_candidates_c(self, seed_E, c, num=100):
        """table caption to find candidate entities"""
        cand = set()
        res = self.__tes.search(query=c, field="catchall", num=num)
        for table_id in res.keys():
            doc = self.__tes.get_doc(table_id)
            labels = doc["_source"]["core_entity_n"]
            cand |= set(labels)
        return cand-set(seed_E)

    def find_candidates_e(self, seed_E, num=100):
        """seed entities to find candidate entities"""
        cand = []
        for entity in seed_E:
            body = self.generate_search_body(item=entity, field="entity_n")
            res = self.__tes.search_complex(body=body, num=num)
            for table_id in res.keys():
                doc = self.__tes.get_doc(table_id)
                labels = doc["_source"]["entity_n"]
                cand += labels
        return set([i for i in cand if i not in seed_E])
    
    def find_core_candidates_e(self, seed_E, num=100):
        """seed entities to find candidate entities"""
        cand = set()
        all_tables = set()
        for entity in seed_E:
            body = self.generate_search_body(item=entity, field="core_entity_n")
            res = self.__tes.search_complex(body=body, num=num)
            all_tables |= set(res.keys())
        for table_id in all_tables:
            doc = self.__tes.get_doc(table_id)
            labels = doc["_source"]["core_entity_n"]
            cand |= set(labels)
        return cand-set(seed_E)

    def generate_search_body(self, item, field):
        """Generate search body"""
        body = {
            "query": {
                "bool": {
                    "must": {
                        "term": {field: item}
                    }
                }
            }
        }
        return body

    def find_candidates_cat(self, seed_E, num=100):  # only category
        """return seed entities' categories"""
        cate_candidates = []
        category = []
        # pdb.set_trace()
        for entity in seed_E:
            entity = "<dbpedia:" + entity + ">"
            try:
                doc = self.__elastic.get_doc(entity)
            except elasticsearch.exceptions.NotFoundError as es:
                print('cannot find category for:', entity)
                continue
            cats = doc.get("_source").get("category_n")
            category += cats

        for cat in set(category):
            body = self.generate_search_body(item=cat, field="category_n")
            res = self.__elastic.search_complex(body=body, num=num)
            cate_candidates += [i[9:-1] if i.startswith('<dbpedia:') else i for i in res.keys() if i not in seed_E]
        return set(cate_candidates)

    def find_core_candidates_cat(self, seed_E, num=100):  # only category
        """return seed entities' categories"""
        cate_candidates = []
        category = []
        # pdb.set_trace()
        for entity in seed_E:
            try:
                doc = self.__elastic.get_doc(entity)
            except elasticsearch.exceptions.NotFoundError as es:
                print('cannot find category for:', entity)
                continue
            cats = doc.get("_source").get("category_n")
            category += cats

        for cat in set(category):
            body = self.generate_search_body(item=cat, field="category_n")
            res = self.__elastic.search_complex(body=body, num=num)
            cate_candidates += [i for i in res.keys() if i not in seed_E]
        return set(cate_candidates)

    def parse(self, text):
        """Put query into a term list for term iteration"""
        stopwords = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him',\
            'his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which',\
            'who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do',\
            'does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about',\
            'against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off',\
            'over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more',\
            'most','other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just','don','should',\
            'now',"i'll","you'll","he'll","she'll","we'll","they'll","i'd","you'd","he'd","she'd","we'd","they'd","i'm","you're","he's","she's",\
            "it's","we're","they're","i've","we've","you've","they've","isn't","aren't","wasn't","weren't","haven't","hasn't","hadn't","don't",\
            "doesn't","didn't","won't","wouldn't","shan't","shouldn't","mustn't","can't","couldn't",'cannot','could',"here's","how's","let's",'ought',\
            "that's","there's","what's","when's","where's","who's","why's",'would']
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
