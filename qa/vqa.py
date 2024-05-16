import requests
import base64

# img_path="../files/img1.jpeg"
# url = "http://34.143.180.202:3389/viscpm"
# resp = requests.post(url, json={
#     # need to modify
#     "image": base64.b64encode(open(img_path, "rb").read()).decode(),
#     "question": "描述一下这张图片",
# })
# resp = resp.json()
# print(resp)

# 假设 img_path 是图片文件的路径
img_path = "/Users/zhangyinhan1/Desktop/code/rag_project/llm_doc_qa/files/img1.jpeg"

# 打开图片文件以二进制读取模式
with open(img_path, "rb") as image_file:
    # 读取文件内容
    image_data = image_file.read()

# 使用 base64 模块的 b64encode 方法将图片数据编码为 Base64 字符串
base64_image_data = base64.b64encode(image_data)

# 使用 decode() 方法将编码后的 bytes 对象转换为字符串
base64_image_str = base64_image_data.decode()

# 现在你可以使用 base64_image_str 变量，它包含了 Base64 编码的图片数据
print(base64_image_str)