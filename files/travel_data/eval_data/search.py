import sys
sys.path.append("/Users/zhangyinhan1/Desktop/code/rag_project/llm_doc_qa/")
from utils.db_client import es_client, milvus_client
from config.config_parser import MILVUS_SIZE, ES_SIZE, MILVUS_THRESHOLD, EMBEDDING_MODEL, COHERE_API_KEY, RERANK_TOP_N
from FlagEmbedding import FlagModel
from collections import OrderedDict

class Search_Engine():
    def __init__(self, query=None):
        self.query = query

    @staticmethod
    def travel_text_embedding(texts):
        model = FlagModel("/Users/zhangyinhan1/Desktop/code/worksace/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
        _ids = []
        sources = []
        contents = []
        embeddings = []
        for i, text in enumerate(texts):
            source = "files/travel_data/all_data.json"
            content = text if len(text)<1000 else text[:1000]
            embedding = model.encode(text)
            _ids.append(i + 1)
            sources.append(source)
            contents.append(content)
            embeddings.append(embedding)
        datas = [_ids, sources, contents, embeddings]
        return datas
    
    def get_milvus_search_result(self):
        # milvus search content
        vectors_to_search = [self.travel_text_embedding(self.query)[-1][0]]
        print(vectors_to_search)
        # 通过嵌入向量相似度获取相似文本
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}, # 取top10
        }
        result = milvus_client.search(vectors_to_search,
                                      "embeddings",
                                      search_params,
                                      limit=MILVUS_SIZE,
                                      output_fields=["text", "source"])
        # filter by similarity score
        return [(_.entity.get('text'), _.entity.get('source')) for _, dist in zip(result[0], result[0].distances) if dist > 0.5]

    def get_es_search_result(self):
        result = []
        # 查询数据(全文搜索)
        dsl = {
        'query': {
            'match': {
                'content': self.query
            }
        },
        "size": 30
    }
        search_result = es_client.search(index='docs', body=dsl)
        if search_result['hits']['hits']:
            result = [_['_source']['content'] for _ in search_result['hits']['hits']]
        return result

    def get_context(self):
        contents = []
        # # 混合检索去重
        # milvus_search_result = self.get_milvus_search_result()
        es_search_result = self.get_es_search_result()
        es_search_result= list(OrderedDict.fromkeys(es_search_result))[:5]
        return es_search_result
        # print(f"milvus 检索结果：{milvus_search_result},结果数量：{len(milvus_search_result)}")
        # print(f"es 检索结果：{es_search_result},结果数量：{len(es_search_result)}")

        # for content_source_tuple in milvus_search_result + es_search_result: # es和mv检索结果
        #     content, source = content_source_tuple
        #     if [content, source] not in contents: # es和mv结果去重
        #         contents.append([content, source])
        # print(f"去重复后的数量：{contents}")
        # return contents


if __name__ == "__main__":
    se = Search_Engine(query="养马岛位于哪个区?")
    res = se.get_context()
    print(res)