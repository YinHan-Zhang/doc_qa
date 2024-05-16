from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_entity_recall,
    context_relevancy
)

from langchain.chat_models import ChatTongyi,ChatZhipuAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from ragas import evaluate
from datasets import Dataset
import pandas as pd
import os
os.environ["DASHSCOPE_API_KEY"] = "sk-db10d05e219d4ba2836874d2b503d1f1"
os.environ["ZHIPUAI_API_KEY"] = "20a2fb44ca962bded2da9b6e5caffbd5.omZe6iDsMaUCM2uf"

df = pd.read_table("/Users/zhangyinhan1/Desktop/code/rag_project/llm_doc_qa/qa/eval_result_last.txt")
del df["contexts"]
df.rename(columns={"truth_contexts":"contexts"},inplace=True)
df["contexts"] = df["contexts"].map(lambda x: eval(x))

dataset = Dataset.from_pandas(df)
dataset = dataset.map(lambda example: {"contexts": [str(context) for context in example["contexts"]]})

print("start to eval....")

result = evaluate(
    dataset, # 传入数据集
    metrics=[
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_entity_recall,
    ],
    # llm=ChatZhipuAI(model="glm-4"),
    llm=ChatTongyi(model="qwen-turbo"),
    embeddings=HuggingFaceBgeEmbeddings(model_name="/Users/zhangyinhan1/Desktop/code/worksace/bge-large-zh-v1.5")
)    

result.to_pandas().to_csv("ragas_eval_res.txt",sep="\t",index=False)


