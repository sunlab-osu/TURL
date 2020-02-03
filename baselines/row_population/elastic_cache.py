"""
elastic_cache
-------------

This is a cache for elastic index stats; a layer between an index and scorer.

@author: Faegheh Hasibi
"""
from elastic import Elastic


class ElasticCache(Elastic):
    def __init__(self, index_name):
        super(ElasticCache, self).__init__(index_name)

        # Cached variables
        self.__num_docs = None
        self.__num_fields = None
        self.__doc_count = {}
        self.__coll_length = {}
        self.__avg_len = {}
        self.__doc_length = {}
        self.__doc_freq = {}
        self.__coll_termfreq = {}

    def __check_cache(self, func, params, var):
        #TODO
        pass

    def num_docs(self):
        """Returns the number of documents in the index."""
        if self.__num_docs is None:
            self.__num_docs = super(ElasticCache, self).num_docs()
        return self.__num_docs

    def num_fields(self):
        """Returns number of fields in the index."""
        if self.__num_fields is None:
            self.__num_fields = super(ElasticCache, self).num_fields()
        return self.__num_fields

    def doc_count(self, field):
        """Returns number of documents with at least one term for the given field."""
        if field not in self.__doc_count:
            self.__doc_count[field] = super(ElasticCache, self).doc_count(field)
        return self.__doc_count[field]

    def coll_length(self, field):
        """Returns length of field in the collection."""
        if field not in self.__coll_length:
            self.__coll_length[field] = super(ElasticCache, self).coll_length(field)
        return self.__coll_length[field]

    def avg_len(self, field):
        """Returns average length of a field in the collection."""
        if field not in self.__avg_len:
            self.__avg_len[field] = super(ElasticCache, self).avg_len(field)
        return self.__avg_len[field]

    def doc_length(self, doc_id, field):
        """Returns length of a field in a document."""
        if doc_id not in self.__doc_length:
            self.__doc_length[doc_id] = {}
        if field not in self.__doc_length[doc_id]:
            self.__doc_length[doc_id][field] = super(ElasticCache, self).doc_length(doc_id, field)
        return self.__doc_length[doc_id][field]

    def doc_freq(self, term, field):
        """Returns document frequency for the given term and field."""
        if field not in self.__doc_freq:
            self.__doc_freq[field] = {}
        if term not in self.__doc_freq[field]:
            self.__doc_freq[field][term] = super(ElasticCache, self).doc_freq(term, field)
        return self.__doc_freq[field][term]

    def coll_term_freq(self, term, field):
        """ Returns collection term frequency for the given field."""
        if field not in self.__coll_termfreq:
            self.__coll_termfreq[field] = {}
        if term not in self.__coll_termfreq[field]:
            self.__coll_termfreq[field][term] = super(ElasticCache, self).coll_term_freq(term, field)
        return self.__coll_termfreq[field][term]
