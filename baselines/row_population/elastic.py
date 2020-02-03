"""
elastic
-------

Tools for working with Elasticsearch.
This class is to be instantiated for each index.

@author: Faegheh Hasibi
@author: Krisztian Balog
"""

from pprint import pprint

from elasticsearch import Elasticsearch
from elasticsearch import helpers

ES_config = {
  "hosts": [
    "localhost:9200"
  ],
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0
  }
}
ELASTIC_HOSTS = ES_config.get("host")
ELASTIC_SETTINGS = ES_config.get("settings")


class Elastic(object):
    FIELD_CATCHALL = "catchall"
    FIELD_ELASTIC_CATCHALL = "_all"
    DOC_TYPE = "doc"  # we don't make use of types
    ANALYZER_STOP_STEM = "english"
    ANALYZER_STOP = "stop_en"
    BM25 = "BM25"
    SIMILARITY = "sim"  # Used when other similarities are used

    def __init__(self, index_name):
        self.__es = Elasticsearch(hosts=ELASTIC_HOSTS)
        self.__index_name = index_name

    @staticmethod
    def analyzed_field(analyzer=ANALYZER_STOP):
        """Returns the mapping for analyzed fields.

        :param analyzer: name of the analyzer; valid options: [ANALYZER_STOP, ANALYZER_STOP_STEM]
        """
        if analyzer not in {Elastic.ANALYZER_STOP, Elastic.ANALYZER_STOP_STEM}:
            print("Error: Analyzer", analyzer, "is not valid.")
            exit(0)
        return {"type": "string",
                "term_vector": "with_positions_offsets",
                "analyzer": analyzer}

    @staticmethod
    def notanalyzed_field():
        """Returns the mapping for not-analyzed fields."""
        return {"type": "string",
                "index": "not_analyzed"}
        # "similarity": Elastic.SIMILARITY}

    def __gen_similarity(self, model, params=None):
        """Gets the custom similarity function."""
        similarity = params if params else {}
        similarity["type"] = model
        return {Elastic.SIMILARITY: similarity}

    def __gen_analyzers(self):
        """Gets custom analyzers.
        We include customized analyzers in the index setting, a field may or may not use it.
        """
        analyzer = {"type": "standard", "stopwords": "_english_"}
        analyzers = {"analyzer": {Elastic.ANALYZER_STOP: analyzer}}
        return analyzers

    def analyze_query(self, query, analyzer=ANALYZER_STOP):
        """Analyzes the query.

        :param query: raw query
        :param analyzer: name of analyzer
        """
        tokens = self.__es.indices.analyze(index=self.__index_name, body=query, analyzer=analyzer)["tokens"]
        query_terms = []
        for t in sorted(tokens, key=lambda x: x["position"]):
            query_terms.append(t["token"])
        return " ".join(query_terms)

    def get_mapping(self):
        """Returns mapping definition for the index."""
        mapping = self.__es.indices.get_mapping(index=self.__index_name, doc_type=self.DOC_TYPE)
        return mapping[self.__index_name]["mappings"][self.DOC_TYPE]["properties"]

    def get_settings(self):
        """Returns index settings."""
        return self.__es.indices.get_settings(index=self.__index_name)[self.__index_name]["settings"]["index"]

    def __update_settings(self, settings):
        """Updates the index settings."""
        self.__es.indices.close(index=self.__index_name)
        self.__es.indices.put_settings(index=self.__index_name, body=settings)
        self.__es.indices.open(index=self.__index_name)
        self.__es.indices.refresh(index=self.__index_name)

    def update_similarity(self, model=BM25, params=None):
        """Updates the similarity function "sim", which is fixed for all index fields.

         The method and param should match elastic settings:
         https://www.elastic.co/guide/en/elasticsearch/reference/2.3/index-modules-similarity.html

        :param model: name of the elastic model
        :param params: dictionary of params based on elastic
        """
        old_similarity = self.get_settings()["similarity"]
        new_similarity = self.__gen_similarity(model, params)
        # We only update the similarity if it is different from the old one.
        # this avoids unnecessary closing of the index
        if old_similarity != new_similarity:
            self.__update_settings({"similarity": new_similarity})

    def delete_index(self):
        """Deletes an index."""
        self.__es.indices.delete(index=self.__index_name)
        print("Index <" + self.__index_name + "> has been deleted.")

    def create_index(self, mappings, model=BM25, model_params=None, force=False):
        """Creates index (if it doesn't exist).

        :param mappings: field mappings
        :param model: name of elastic search similarity
        :param model_params: name of elastic search similarity
        :param force: forces index creation (overwrites if already exists)
        """
        if self.__es.indices.exists(self.__index_name):
            if force:
                self.delete_index()
            else:
                print("Index already exists. No changes were made.")
                return

        # sets general elastic settings
        body = ELASTIC_SETTINGS

        # sets the global index settings
        # number of shards should be always set to 1; otherwise the stats would not be correct
        body["settings"] = {"analysis": self.__gen_analyzers(),
                            "index": {"number_of_shards": 1,
                                      "number_of_replicas": 0},
                            }

        # sets similarity function
        # If model is not BM25, a similarity module with the given model and params is defined
        if model != Elastic.BM25:
            body["settings"]["similarity"] = self.__gen_similarity(model, model_params)
        sim = model if model == Elastic.BM25 else Elastic.SIMILARITY
        for mapping in mappings.values():
            mapping["similarity"] = sim

        # sets the field mappings
        body["mappings"] = {self.DOC_TYPE: {"properties": mappings}}

        # creates the index
        self.__es.indices.create(index=self.__index_name, body=body)
        pprint(body)
        print("New index <" + self.__index_name + "> is created.")

    def add_docs_bulk(self, docs):
        """Adds a set of documents to the index in a bulk.

        :param docs: dictionary {doc_id: doc}
        """
        actions = []
        for doc_id, doc in docs.items():
            action = {
                "_index": self.__index_name,
                "_type": self.DOC_TYPE,
                "_id": doc_id,
                "_source": doc
            }
            actions.append(action)

        if len(actions) > 0:
            helpers.bulk(self.__es, actions)

    def add_doc(self, doc_id, contents):
        """Adds a document with the specified contents to the index.

        :param doc_id: document ID
        :param contents: content of document
        """
        self.__es.index(index=self.__index_name, doc_type=self.DOC_TYPE, id=doc_id, body=contents)

    def get_doc(self, doc_id, fields=None, source=True):
        """Gets a document from the index based on its ID.

        :param doc_id: document ID
        :param fields: list of fields to return (default: all)
        :param source: return document source as well (default: yes)
        """
        return self.__es.get(index=self.__index_name, doc_type=self.DOC_TYPE, id=doc_id, _source=source)

    def del_doc(self, doc_id):
        """Gets a document from the index based on its ID.

        :param doc_id: document ID
        :param fields: list of fields to return (default: all)
        :param source: return document source as well (default: yes)
        """
        return self.__es.delete(index=self.__index_name, doc_type=self.DOC_TYPE, id=doc_id)

    def search(self, query, field, num=100, fields_return="", start=0):
        """Searches in a given field using the similarity method configured in the index for that field.

        :param query: query string
        :param field: field to search in
        :param num: number of hits to return (default: 100)
        :param fields_return: additional document fields to be returned
        :param start: starting offset (default: 0)
        :return: dictionary of document IDs with scores
        """
        hits = self.__es.search(index=self.__index_name, q=query, df=field, _source=False, size=num,
                                from_=start)["hits"]["hits"]
        results = {}
        for hit in hits:
            results[hit["_id"]] = hit["_score"]
        return results

    def estimate_number(self, query):
        """Search body, return the number of hits containg body"""
        try:
            return self.__es.search(index=self.__index_name, q = query, _source=False, size=1,
                                from_=0)["hits"]["total"]
        except:
            return 0

    def search_complex(self, body, fields_return="", num=100, start=0):
        """Searches in a given field using the similarity method configured in the index for that field.

        :param body: query body
        :param field: field to search in
        :param num: number of hits to return (default: 100)
        :param fields_return: additional document fields to be returned
        :param start: starting offset (default: 0)
        :return: dictionary of document IDs with scores
        """
        hits = self.__es.search(index=self.__index_name, body=body, _source=False, size=num,
                                from_=start)["hits"]["hits"]
        results = {}
        for hit in hits:
            results[hit["_id"]] = hit["_score"]
        return results

    def get_ids(self, body):
        """Search body, return the number of hits containg body"""
        try:
            return self.search_complex(body).keys()
        except:
            return 0

    def estimate_number_complex(self, body):
        """Search body, return the number of hits containg body"""
        try:
            return self.__es.search(index=self.__index_name, body=body, _source=False, size=1,
                                from_=0)["hits"]["total"]
        except:
            return 0

    def get_field_stats(self, field):
        """Returns stats of the given field."""
        return self.__es.field_stats(index=self.__index_name, fields=[field])["indices"]["_all"]["fields"][field]

    def get_fields(self):
        """Returns name of fields in the index."""
        return list(self.get_mapping().keys())

    # =========================================
    # ================= Stats =================
    # =========================================
    def __get_termvector(self, doc_id, field, term_stats=False):
        """Returns a term vector for a given document field, including global field and term statistics.
        Term stats can have a serious performance impact; should be set to true only if it is needed!

        :param doc_id: document ID
        :param field: field name
        """
        tv = self.__es.termvectors(index=self.__index_name, doc_type=self.DOC_TYPE, id=doc_id, fields=field,
                                   term_statistics=term_stats)
        return tv.get("term_vectors", {}).get(field, {}).get("terms", {})

    def __get_coll_termvector(self, term, field):
        """Returns a term vector containing collection stats of a term."""
        hits = self.search(term, field, num=1)
        doc_id = next(iter(hits.keys())) if len(hits) > 0 else None
        return self.__get_termvector(doc_id, field, term_stats=True) if doc_id else {}

    def num_docs(self):
        """Returns the number of documents in the index."""
        return self.__es.count(index=self.__index_name, doc_type=self.DOC_TYPE)["count"]

    def num_fields(self):
        """Returns number of fields in the index."""
        return len(self.get_mapping())

    def doc_count(self, field):
        """Returns number of documents with at least one term for the given field."""
        return self.get_field_stats(field)["doc_count"]

    def coll_length(self, field):
        """Returns length of field in the collection."""
        return self.get_field_stats(field)["sum_total_term_freq"]

    def avg_len(self, field):
        """Returns average length of a field in the collection."""
        return self.coll_length(field) / self.doc_count(field)

    def doc_length(self, doc_id, field):
        """Returns length of a field in a document."""
        return sum(self.term_freqs(doc_id, field).values())

    def doc_freq(self, term, field):
        """Returns document frequency for the given term and field."""
        tv = self.__get_coll_termvector(term, field)
        return tv.get(term, {}).get("doc_freq", 0)

    def coll_term_freq(self, term, field):
        """ Returns collection term frequency for the given field."""
        tv = self.__get_coll_termvector(term, field)
        return tv.get(term, {}).get("ttf", 0)

    def term_freqs(self, doc_id, field):
        """Returns term frequencies for a given document and field.

        :return dictionary of terms with their frequencies; {doc_id: freq, ...}
        """
        tv = self.__get_termvector(doc_id, field)
        term_freqs = {}
        for term, val in tv.items():
            term_freqs[term] = val["term_freq"]
        return term_freqs

    def term_freq(self, doc_id, field, term):
        """Returns frequency of a term in a given document and field."""
        return self.term_freqs(doc_id, field).get(term, 0)


if __name__ == "__main__":
    field = "content"
    term = "gonna"
    doc_id = 4

    es = Elastic("toy_index")
    pprint(es.search("gonna", "content"))

    print("================= Stats =================")
    print("[FIELD]: %s [TERM]: %s" % (field, term))
    print("- Number of documents: %d" % es.num_docs())
    print("- Number of fields: %d" % es.num_fields())
    print("- Document count: %d" % es.doc_count(field))
    print("- Collection length: %d" % es.coll_length(field))
    print("- Average length: %.2f" % es.avg_len(field))
    print("- Document length: %d" % es.doc_length(doc_id, field))
    print("- Number of fields: %d" % es.num_fields())
    print("- Document frequency: %d" % es.doc_freq(term, field))
    print("- Collection frequency: %d" % es.coll_term_freq(term, field))
    print("- Term frequencies:")
    pprint(es.term_freqs(doc_id, field))
    person_id = "you"
    # search doc containing person_id
    body = {
        "query": {
            "bool": {
                "must": {
                    "term": {"content": person_id}
                }
            }
        }
    }
    # search docs containing both person_id and a(analyzed)
    a = es.analyze_query("me")
    body = {"query": {
        "bool": {
            "must": [
                {
                    "match": {"content": person_id}
                },
                {
                    "match_phrase": {"content": a}
                }
            ]
        }
    }}
    print(es.search_complex(body, "content"))
    print(es.term_freqs(1, "content"))

    # pprint.pprint(es.get_termvector("<dbpedia:Category:People_of_Canadian_descent>", "title"))
    # pprint.pprint(es.search("people", "title", fields_return="title"))
