# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: file_parser.py
# @time: 2023/9/8 18:01
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain.schema import Document
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, SeleniumURLLoader,PDFPlumberLoader
import sys
sys.path.append("/Users/zhangyinhan1/Desktop/code/rag/llm_4_doc_qa")
from utils.logger import logger


class FileParser(object):
    def __init__(self, file_path, file_content=""):
        self.file_path = file_path
        self.file_content = file_content

    def string_loader(self):
        documents = Document(page_content=self.file_content, metadata={"source": self.file_path})
        return [documents]

    def txt_loader(self):
        documents = TextLoader(self.file_path, encoding='utf-8').load()
        return documents

    def pdf_loader(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load_and_split()
        return documents
    
    def pdf_loader_v2(self):
        loader = PDFPlumberLoader(self.file_path)
        documents = loader.load_and_split()
        return documents
        

    def docx_loader(self):
        loader = Docx2txtLoader(self.file_path)
        documents = loader.load()
        return documents

    def url_loader(self):
        loader = SeleniumURLLoader(urls=[self.file_path])
        documents = loader.load()
        return documents

    def parse(self):
        logger.info(f'parse file: {self.file_path}')
        if self.file_content:
            return self.string_loader(), 'string'
        else:
            if self.file_path.endswith(".txt"):
                return self.txt_loader(), 'txt'
            elif self.file_path.endswith(".pdf"):
                return self.pdf_loader_v2(), 'pdf'
            elif self.file_path.endswith(".docx"):
                return self.docx_loader(), 'docx'
            elif "http" in self.file_path:
                return self.url_loader(), 'url'
            else:
                logger.error("unsupported document type!")
                return [], ''


if __name__ == '__main__':
    # txt_file_path = "../files/gdp.txt"
    # content = FileParser(txt_file_path).parse()
    # print(content)

    pdf_file_path = "../files/swan.pdf"
    content = FileParser(pdf_file_path).parse()
    print(content)

    # docx_file_path = "../files/haicaihua.docx"
    # content = FileParser(docx_file_path).parse()
    # print(content)

    # url = "https://gaokao.xdf.cn/202303/12967078.html"
    # url = "https://www.hntv.tv/50rd/article/1/1700396378818207745?v=1.0"
    # content = FileParser(url).parse()
    # print(content)
