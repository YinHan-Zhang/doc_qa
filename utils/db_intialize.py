# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: db_intialize.py
# @time: 2023/7/22 12:47
from pymilvus import (
    connections,
    Milvus,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from elasticsearch import Elasticsearch

# 连接Elasticsearch
es_client = Elasticsearch([{'host': 'localhost', 'port': 9200}])

settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "ik_smart": {
                    "type": "ik_smart"
                }
            }
        }
    }
}
# 创建新的ES index
mapping = {
    'properties': {
        'source': {
            'type': 'text',
            'fields': {
                "keyword": {
                    "type": "keyword",
                    "ignore_above": 256
                }
            }
        },
        'cont_id': {
            'type': 'integer'
        },
        'content': {
            'type': 'text',
            'analyzer': 'ik_smart',
            'search_analyzer': 'ik_smart'
        },
        'file_type': {
            'type': 'keyword'
        },
        "insert_time": {
            "type": "date",
            "format": "yyyy-MM-dd HH:mm:ss"
        }
    }
}
def es_get_index(es_client):
    # 获取所有索引的名称
    indices = es_client.indices.get_alias("*")

    # 打印所有索引的名称
    for index in indices:
        print(index)

    # 查看索引详细信息
    index_info = es_client.indices.get(index="docs")
    print(index_info)

def es_delete_index(es_client):
    # 删除索引
    es_client.indices.delete(index="docs")
    print("索引删除成功！")

def es_create_index(es_client):
    # 创建索引并指定 settings 和 mapping
    es_client.indices.create(index='docs', body=settings, ignore=400)
    # 索引创建成功后，应用 mapping
    result = es_client.indices.put_mapping(index='docs', body=mapping)
    print("索引创建成功！",result)



def mv_create_index():
    # 连接milvus
    connections.connect("default", host="localhost", port="19530")
    # # 创建一个collection
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024)   # 5120 for baichuan, 1536 for openai, 1024 for glm4 and bge, 300 for paddle
    ]
    schema = CollectionSchema(fields, "vector db for travel docs qa")

    docs_milvus = Collection("travel_docs_qa", schema)

    # # 在embeddings字段创建索引
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 128},
    }
    docs_milvus.create_index("embeddings", index)

def mv_delete_index():
    # 连接milvus
    client = Milvus(host='localhost', port='19530')

    # 如果集合已存在，先删除它
    if "travel_docs_qa" in client.list_collections():
        client.drop_collection("travel_docs_qa")
        print("已删除集合:", "travel_docs_qa")


    client.close()

if __name__ == "__main__":
    es_create_index(es_client)
    # es_delete_index(es_client)
    # es_get_index(es_client)
    # mv_create_index()
    # mv_delete_index()
