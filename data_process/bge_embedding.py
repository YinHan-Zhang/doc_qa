from flag_models import FlagModel
sentences_1 = "样例数据-1"
model = FlagModel("/Users/zhangyinhan1/Desktop/code/worksace/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
embeddings_1 = model.encode(sentences_1)
print(embeddings_1, len(embeddings_1))