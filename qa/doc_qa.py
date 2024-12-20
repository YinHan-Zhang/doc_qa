# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: doc_qa.py
# @time: 2023/7/22 14:06
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# import cohere
import sys
sys.path.append("/Users/zhangyinhan1/Desktop/code/rag_project/llm_doc_qa")
from utils.db_client import es_client, milvus_client
from data_process.data_processor import get_text_embedding
from common.llm_chat_api import chat_completion
from utils.logger import logger
from config.config_parser import MILVUS_SIZE, ES_SIZE, MILVUS_THRESHOLD, EMBEDDING_MODEL, COHERE_API_KEY, RERANK_TOP_N
from FlagEmbedding import FlagReranker,FlagModel
import numpy as np
import dashscope
import os
import pandas as pd
dashscope.api_key = "sk-db10d05e219d4ba2836874d2b503d1f1"

# 文档问答
class DocQA(object):
    def __init__(self, query=None, img=None):
        self.query = query
        self.img = img
        self.rerank_model = FlagReranker("/Users/zhangyinhan1/Desktop/code/worksace/bge_rerank",use_fp16=True)
        self.embeddings_model = FlagModel("/Users/zhangyinhan1/Desktop/code/worksace/bge-large-zh-v1.5", query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：")
    def set_query(self,query):
        self.query = query
        return self.query
    def get_milvus_search_result(self):
        
        # milvus search content
        # vectors_to_search = [get_text_embedding(self.query, EMBEDDING_MODEL)]
        vectors_to_search = [self.embeddings_model.encode(self.query)]
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
        return [(_.entity.get('text'), _.entity.get('source')) for _, dist in zip(result[0], result[0].distances) if dist > MILVUS_THRESHOLD]

    def get_es_search_result(self):
        result = []
        # 查询数据(全文搜索)
        dsl = {
            'query': {
                'match': {
                    'content': self.query
                }
            },
            "size": 10
        }
        search_result = es_client.search(index='docs', body=dsl)
        if search_result['hits']['hits']:
            result = [(_['_source']['content'], _['_source']['source']) for _ in search_result['hits']['hits']]
        return result

    def get_context(self):
        contents = []
        # 去重
        milvus_search_result = self.get_milvus_search_result()
        es_search_result = self.get_es_search_result()
        logger.info(f"milvus search result nums:{len(milvus_search_result)}, es search result nums: {len(es_search_result)}")
        for content_source_tuple in milvus_search_result + es_search_result:
            content, source = content_source_tuple
            if [content, source] not in contents:
                contents.append([content, source])
        return contents

    def rerank_api(self):
        before_rerank_contents = self.get_context()
        # cohere_client = cohere.Client(COHERE_API_KEY)
        # docs, sources = [_[0] for _ in before_rerank_contents], [_[1] for _ in before_rerank_contents]
        # results = cohere_client.rerank(model="rerank-multilingual-v2.0",
        #                                query=self.query,
        #                                documents=docs,
        #                                top_n=RERANK_TOP_N)
        # after_rerank_contents = []
        # for hit in results:
        #     after_rerank_contents.append([hit.document['text'], sources[hit.index]])
        #     logger.info(f"score: {hit.relevance_score}, query: {self.query}, text: {hit.document['text']}")
        # return after_rerank_contents
        return before_rerank_contents
    
    def rerank(self):
        before_rerank_contents = self.get_context()
        docs, sources = [_[0] for _ in before_rerank_contents], [_[1] for _ in before_rerank_contents]
        # 查看rerank之前的结果
        # logger.info(docs)  
        
        # logger.info("load rerank model success!")
        pairs = [[self.query, doc] for doc in docs]
        scores = self.rerank_model.compute_score(pairs)
        after_rerank_contents = []
        ranked_indices = np.argsort(-np.array(scores))
        for i in ranked_indices[:3]: # 取top5
            after_rerank_contents.append([docs[i], sources[i]])
        return after_rerank_contents
    
    def get_qa_prompt(self):
        # 建立prompt
        prefix = "<文本片段>:\n\n"
        suffix = f"\n<问题>: {self.query}\n<回答>: "
        prompt = []
        contexts = []
        contexts_list = []
        sources = []
        for i, text_source_tuple in enumerate(self.rerank()):
            text, source = text_source_tuple
            prompt.append(f"{i+1}: {text}\n")
            contexts.append(f"<{i+1}>: {text}")
            contexts_list.append(text)
            sources.append(f"<{i+1}>: {source}")
        """
        ## 优化部分
        ## denoise
        prompts = self.prompt_denoise(contexts_list)
        try:
            left_index , right_index = prompts.find("["), prompts.rfind("]")
            prompt = [f"{i+1}: {text}\n" for i,j in enumerate(eval(prompts[left_index:right_index+1]))]
        except Exception as e:
            logger.info(e)
            prompt = contexts_list

        # compresion
        if len("".join(prompt))>500:
            compress_prompt = self.prompt_compression(prompt)
            qa_chain_prompt = prefix + compress_prompt + suffix
        else:
            qa_chain_prompt = prefix + ''.join(prompt) + suffix
        """
        qa_chain_prompt = prefix + ''.join(prompt) + suffix
        contexts, sources = "\n\n".join(contexts), "\n\n".join(sources)
        logger.info(qa_chain_prompt)
        return qa_chain_prompt, contexts, sources, contexts_list
        
    def qwen_model(self, input, model_type):
            resp = dashscope.Generation.call(
                model = model_type,
                prompt = input
            )
            return resp.output.text
    
    def prompt_denoise(self,context=None):
        # qa_chain_prompt, contexts, sources, contexts_list = self.get_qa_prompt()
        model_type = "qwen-max"
        prompt = f"你是一名资深的导游。请根据问题和答案列表的相关性，去除掉答案列表中无关的答案，最终返回与问题相关的新的列表。\n问题:{self.query}\n答案列表:{str(context)}\n去除与问题的无关信息后的答案列表:"
        res = self.qwen_model(prompt, model_type)
        return res
    
    def prompt_compression(self,context=None):
        # qa_chain_prompt, contexts, sources, contexts_list = self.get_qa_prompt()
        model_type = "qwen-turbo"
        prompt = f"你是一名资深的导游。请根据问题和答案列表的相关性进行信息压缩。\n问题:{self.query}\n答案列表:{str(context)}\n压缩后的信息:"
        res = self.qwen_model(prompt, model_type)
        return res

    def answer(self, model_name, img_path=None):
        message, contexts, sources, contexts_list = self.get_qa_prompt()
        result = chat_completion(message, model_name, img_path=self.img)
        return result, contexts, sources


if __name__ == '__main__':
    # test_model_name = "glm-4"
    # test_model_name = "glm-4v"
    # question = "图片中描述的是什么？"
    # img_path = "/Users/zhangyinhan1/Desktop/code/rag_project/llm_doc_qa/files/img1.jpeg"
    # question = '美国人什么时候登上月球的？'
    # question = '戚发轫的职务是什么？'
    # question = '你知道格里芬的职务吗？'
    # question = '格里芬发表演说时讲了什么？'
    # question = '五一去厦门有什么好玩的嘛？'
    # qa = DocQA(question)
    # print(qa.prompt_denoise())
    # print(qa.prompt_compression())
    # reply = DocQA(question).answer(model_name=test_model_name)
    # reply = DocQA(question).answer(model_name=test_model_name, img_path=img_path)
    # print(reply)

    import pandas as pd
    eval_data = pd.read_csv("/Users/zhangyinhan1/Desktop/code/rag_project/llm_doc_qa/files/travel_data/eval_data/eval_data_qg_csv.csv")
    eval_result = []
    qa = DocQA()
    for idx, row in eval_data.iterrows():
        # if idx < 150:
        #     continue
        if idx % 50 == 0:
            print(f"{idx} is done !")
            pd.DataFrame(eval_result).to_csv(f"eval_result_{idx}.txt",sep="\t",index=False)
        
        query = qa.set_query(row["question"])
        result, contexts, sources = qa.answer(model_name="glm-4")
        eval_result.append(
            {
                "question": query,
                "ground_truth" : row["ground_truth"],
                "truth_contexts": row["contexts"],
                "contexts": contexts,
                "answer": result
            }
        )
    pd.DataFrame(eval_result).to_csv("eval_result_last.txt",sep="\t",index=False)


