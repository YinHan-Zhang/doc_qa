# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: server_gradio.py
# @time: 2023/9/8 23:11
import json
import re
import gradio as gr
from config.config_parser import MODEL_NAME_LIST
from qa.doc_qa import DocQA
from qa.doc_qa_evaluation import DocQAEvaluation
import base64
import PIL

# 对话历史记录，用于存储每一轮的问答
dialogue_history = []

def image_to_base64(image_file):
    """
    将上传的图片文件转换为 Base64 编码的字符串。
    """
    if image_file is None:
        return None
    
    img_path = image_file.name
    # 使用 base64 库将字节流编码为 Base64 字符串
    base64_str = base64.b64encode(open(img_path, "rb").read()).decode('utf-8')
    return base64_str

def img_process(image_file):
    img = PIL.Image.open(image_file.name)
    # 调整图像尺寸
    # 这里设置新的尺寸为（宽度：300，高度：按比例缩放）
    # 注意：img.size是一个元组(width, height)
    base_width = 512
    w_percent = (base_width / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((base_width, h_size), PIL.Image.Resampling.LANCZOS)
    return img

def document_qa(model_list, is_show_reference, question, image_file):
    global dialogue_history
    output = []
    contexts_output = []
    # print(image_file)
    # 如果有上传的图片，将其转换为 Base64 编码字符串
    if image_file is not None:
        img_path = image_to_base64(image_file)
        # img = img_process(image_file)
        # print(img_path)
    else:
        img_path = None

    for model_name in model_list:
        reply, contexts, sources = DocQA(question, img=img_path).answer(model_name)
        if is_show_reference:
            metric_result = DocQAEvaluation(question=question, answer=reply, context=contexts).evaluate()
            metric = "\n".join([f"{k}: {v}" for k, v in metric_result.items()])
            output.append([model_name, question, reply, contexts, sources, metric])
            contexts_output.append(contexts)
        else:
            output.append([model_name, question, reply, "", "", ""])
            contexts_output.append("")
    # 更新对话历史记录
    for model_name, reply in zip(model_list, output):
        dialogue_history.append(("User: ", question))
        dialogue_history.append((f"Model ({model_name}): ", reply[2]))  # 示例，根据实际情况调整
    return output, dialogue_history, "\n\n".join(contexts_output), PIL.Image.open(image_file.name)


# find most like sentence
def find_most_like_sentence(answer, candidates):
    similarity_list = []
    for candidate in candidates:
        s1 = set(answer)
        s2 = set(candidate)
        similarity = len(s1 & s2) / len(s1 | s2)
        similarity_list.append(similarity)

    flag, max_num = 0, 0
    for i in range(len(similarity_list)):
        if similarity_list[i] > max_num:
            flag = i
            max_num = similarity_list[i]

    return flag


def highlight(df):
    reply = df.iloc[0, 2]
    contexts = df.iloc[0, 3]
    sents = [_ for _ in re.split(r"<\d>:", contexts) if _]
    flag = find_most_like_sentence(reply, sents)
    # for highlight
    compare = []
    for i, sent in enumerate(sents):
        compare.append((f"<{i+1}>", "other"))
        if i != flag:

            for char in sent:
                compare.append((char, "other"))
        else:
            for char in sent:
                if char in reply:
                    compare.append((char, "same"))
                else:
                    compare.append((char, "other"))

    return compare




# Gradio 应用定义
with gr.Blocks() as demo:
    # 定义组件
    models = gr.CheckboxGroup(choices=MODEL_NAME_LIST,
                value=MODEL_NAME_LIST[1],
                label="LLMs")
    show_reference = gr.Checkbox(label="Show Answer Reference")
    history_output = gr.Chatbot(label="Dialogue History")
    q = gr.Textbox(lines=3, placeholder="Your question ...", label="Question")
    image_input = gr.File(label="Upload Image", file_types=["image"]) # 图片上传组件
    
    greet_btn = gr.Button("Submit")
    answer = gr.DataFrame(label='Answer', headers=["model", "question", "answer", "reference", "source", "metric"], wrap=True)
    contexts_output = gr.Text(label="Retrieve Konwledage")

    image_display = gr.Image(label="Uploaded Image Display")

    greet_btn.click(fn=document_qa, 
                    inputs=[models, show_reference, q, image_input], 
                    outputs=[answer, history_output, contexts_output, image_display])

   

# 启动应用
demo.launch()
    

