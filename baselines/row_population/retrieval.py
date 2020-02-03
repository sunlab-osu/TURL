"""
retrieval
---------

Console application for general-purpose retrieval.

* *First pass*: get top ``N`` documents using Elastic's default retrieval method (based on the catch-all content field)
* *Second pass*: perform (expensive) scoring of the top ``N`` documents using the Scorer class

@author: Krisztian Balog
@author: Faegheh Hasibi
"""
import argparse
import json

import sys
from pprint import pprint

from Population.elastic import Elastic
from Population.elastic_cache import ElasticCache
from Population.scorer import Scorer, ScorerLM
#from file_utils import FileUtils


class Retrieval(object):
    """Loads config file, checks params, and sets default values.

    :param config: retrieval config (JSON config file or a dictionary) of the shape:

    ::

        {
            "index_name": name of the index,
            "first_pass": {
                "num_docs": number of documents in first-pass scoring (default: 10000)
                "field": field used in first pass retrieval (default: Elastic.FIELD_CATCHALL)
            },
            "second_pass": {
                "num_docs": number of documents to return (default: 100)
                "field": field name (for single field models; e.g., LM, SDM)
                "fields": list of fields (for multiple field models; e.g., MLM, PRMS)
                "field_weights": dictionary with fields and corresponding weights (for MLM and FSDM)
                "model": name of retrieval model; accepted values: [lm, mlm, prms, sdm, fsdm] (default: lm)
                "smoothing_method": accepted values: [jm, dirichlet] (default: dirichet)
                "smoothing_param": value of lambda or mu accepted values: [float or "avg_len"],
                                    (jm default: 0.1, dirichlet default: 2000)
            },
            "query_file": name of query file (JSON),
            "output_file": name of output file,
            "run_id": run id for TREC output
        }

    """
    FIELDED_MODELS = {"mlm", "prms", "fsdm"}
    LM_MODELS = {"lm", "mlm", "prms", "sdm", "fsdm"}

    def __init__(self, config):
        self.__check_config(config)
        pprint(config)
        self.__config = config
        self.__index_name = config["index_name"]
        self.__first_pass_num_docs = config["first_pass"]["num_docs"]
        self.__first_pass_field = config["first_pass"]["field"]
        self.__first_pass_model = config["first_pass"]["model"]
        self.__second_pass = config.get("second_pass", None)
        self.__second_pass_model = config.get("second_pass", {}).get("model", None)
        self.__second_pass_num_docs = config.get("second_pass", {}).get("num_docs", None)
        self.__query_file = config.get("query_file", None)
        self.__output_file = config.get("output_file", None)
        self.__run_id = config.get("run_id", self.__second_pass_model)

        self.__elastic = ElasticCache(self.__index_name)

    @staticmethod
    def __check_config(config):
        """Checks config parameters and sets default values."""
        try:
            if "index_name" not in config:
                raise Exception("index_dir is missing")
            # Checks first pass parameters
            if "first_pass" not in config:
                config["first_pass"] = {}
            if "num_docs" not in config["first_pass"]:
                config["first_pass"]["num_docs"] = 1000
            if "field" not in config["first_pass"]:
                config["first_pass"]["field"] = Elastic.FIELD_CATCHALL
            if "model" not in config["first_pass"]:
                config["first_pass"]["model"] = Elastic.BM25
            # todo: set default params for "params" (from elastic search)

            # Checks second pass parameters
            if "second_pass" in config:
                if "num_docs" not in config["second_pass"]:
                    config["second_pass"]["num_docs"] = 100
                if "field" not in config["second_pass"]:
                    config["second_pass"]["field"] = Elastic.FIELD_CATCHALL
                if "model" not in config["second_pass"]:
                    config["second_pass"]["model"] = "lm"
                if config["second_pass"]["model"] in Retrieval.LM_MODELS:
                    if "smoothing_method" not in config["second_pass"]:
                        config["second_pass"]["smoothing_method"] = ScorerLM.DIRICHLET
                    if "smoothing_param" not in config["second_pass"]:
                        if config["second_pass"]["smoothing_method"] == ScorerLM.DIRICHLET:
                            config["second_pass"]["smoothing_param"] = 2000
                        elif config["second_pass"]["smoothing_method"] == ScorerLM.JM:
                            config["second_pass"]["smoothing_param"] = 0.1
                        else:
                            raise Exception("Smoothing method is not supported.")
                # todo: set default params for "fields" (for MLM, PRMS, etc.)
        except Exception as e:
            print("Error in config file: ", e)
            sys.exit(1)

    def _first_pass_scoring(self, analyzed_query):
        """Returns first-pass scoring of documents.

        :param analyzed_query: analyzed query
        :return: RetrievalResults object
        """
        print("\tFirst pass scoring... ", )
        # todo: add support for other similarities
        # self.__elastic.update_similarity(self.__first_pass_model, self.__first_pass_model_params)
        res1 = self.__elastic.search(analyzed_query, self.__first_pass_field, num=self.__first_pass_num_docs)
        return res1

    def _second_pass_scoring(self, res1, scorer):
        """Returns second-pass scoring of documents.

        :param res1: first pass results
        :param scorer: scorer object
        :return: RetrievalResults object
        """
        print("\tSecond pass scoring... ", )
        res2 = {}
        for doc_id in res1.keys():
            res2[doc_id] = scorer.score_doc(doc_id)
        print("done")
        return res2

    def retrieve(self, query):
        """Scores documents for the given query."""
        query = self.__elastic.analyze_query(query)
        # 1st pass retrieval
        res1 = self._first_pass_scoring(query)
        if not self.__second_pass:
            return res1

        # 2nd pass retrieval
        scorer = Scorer.get_scorer(self.__elastic, self.__second_pass_model, query, self.__second_pass)
        res2 = self._second_pass_scoring(res1, scorer)
        return res2

    def batch_retrieval(self):
        """Scores queries in a batch and outputs results."""
        queries = json.load(open(self.__query_file))

        # sets the numbers of documents in the trec file
        max_rank = self.__second_pass_num_docs if self.__second_pass else self.__first_pass_num_docs

        # init output file
        open(self.__output_file, "w").write("")
        out = open(self.__output_file, "w")

        # retrieves documents
        for query_id in sorted(queries):
            print("scoring [" + query_id + "] " + queries[query_id])
            results = self.retrieve(queries[query_id])
            out.write(self.trec_format(results, query_id, max_rank))
        out.close()
        print("Output file:", self.__output_file)

    def trec_format(self, results, query_id, max_rank=100):
        """Outputs results in TREC format"""
        out_str = ""
        rank = 1
        for doc_id, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            if rank > max_rank:
                break
            out_str += query_id + "\tQ0\t" + doc_id + "\t" + str(rank) + "\t" + str(score) + "\t" + self.__run_id + "\n"
            rank += 1
        return out_str


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file", type=str)
    args = parser.parse_args()
    return args


def main():
    dbpedia_config = {"index_name": "dbpedia_2015_10",
                      "first_pass": {
                          "num_docs": 1000
                      },
                      "second_pass": {
                          "model": "lm",
                          "num_docs": 1000,
                          "smoothing_method": "dirichlet",
                          "smoothing_param": 2000,
                          "field_weights": {"catchall": 0.4, "related_entity_names": 0.2, "categories": 0.4}
                      },
                      # "query_file": "data/queries/dbpedia-entity.json",
                      # "output_file": "output/mlm_tc.txt",
                      "run_id": "mlm_tc"
                      }
    r = Retrieval(dbpedia_config)
    pprint(r.retrieve("gonna"))


if __name__ == "__main__":
    main()
