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

# ����Elasticsearch
es_client = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# �����µ�ES index
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

es_client.indices.create(index='docs', ignore=400)
result = es_client.indices.put_mapping(index='docs', body=mapping)
print(result)

# ����milvus
# client = Milvus(host='localhost', port='19530')
connections.connect("default", host="localhost", port="19530")

# ��������Ѵ��ڣ���ɾ����
# if "docs_qa" in client.list_collections():
#     client.drop_collection("docs_qa")
#     print("��ɾ������:", "docs_qa")

# ����һ��collection
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=100),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1024)   # 5120 for baichuan, 1536 for openai, 1024 for glm4
]
schema = CollectionSchema(fields, "vector db for docs qa")

docs_milvus = Collection("docs_qa", schema)

# ��embeddings�ֶδ�������
index = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 128},
}
docs_milvus.create_index("embeddings", index)
# client.close()