class PDFPlumberParser(BaseBlobParser):
    """Parse `PDF` with `PDFPlumber`."""

    def __init__(
        self,
        text_kwargs: Optional[Mapping[str, Any]] = None,
        dedupe: bool = False,
        extract_images: bool = False,
    ) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe
        self.extract_images = extract_images

    # def lazy_parse(self, blob: Blob) -> Iterator[Document]:
    #     """Lazily parse the blob."""
    #     import pdfplumber

    #     with blob.as_bytes_io() as file_path:
    #         doc = pdfplumber.open(file_path)  # open document

    #         yield from [
    #             Document(
    #                 page_content=self._process_page_content(page)
    #                 + "\n"
    #                 + self._extract_images_from_page(page),
    #                 metadata=dict(
    #                     {
    #                         "source": blob.source,
    #                         "file_path": blob.source,
    #                         "page": page.page_number - 1,
    #                         "total_pages": len(doc.pages),
    #                     },
    #                     **{
    #                         k: doc.metadata[k]
    #                         for k in doc.metadata
    #                         if type(doc.metadata[k]) in [str, int]
    #                     },
    #                 ),
    #             )
    #             for page in doc.pages
    #         ]
    

    def _table2md(self,table):
        
        col = []
        for i in table[0]:
            
            if i is not None:
                col.append(i)
                pre = i
            else:
                col.append(pre)
            
        df = pd.DataFrame(table[1:], columns=col)
        df.replace(np.nan, '', inplace=True)
        for i in range(len(df.columns)):
            df.iloc[:,i]=df.iloc[:,i].str.replace("\n", '')
        return df

    def _judge_next(self,pdf, cnt, df):
        next_page = pdf.pages[cnt]
        next_page_tables = next_page.find_tables()  # 如果是跨页表格，当前页只有一个表格  
        if next_page_tables == []:
            return None, False
        # print("跨页合并 ，当前page:",cnt)
        min_x, min_y, max_x, max_y = next_page.bbox[0], next_page.bbox[1], next_page.bbox[2], next_page.bbox[3]
        # print("table坐标：",next_page_tables)
        x1,y1,x2,y2 = next_page_tables[0].bbox
        df_ = self._table2md(next_page.within_bbox((0,0,max_x,max_y)).extract_table())

        if  df.columns.tolist() == df_.columns.tolist():
            # print(df.info())
            # print("---"*100)
            # print(df_.info())
            ps.append(cnt)
            new_df = pd.concat([df, df_],axis=0)
            # print(new_df.info())
            return new_df, True 
        else:
            return None, False
        
    def _continue_table(self,pdf, p, table):
        
        cnt = p
        df = self._table2md(table) # 第一页结尾的表格
        # print("第一页结尾表格：")
        # print(df.info())
        flag = True
        while flag:
            cnt+=1
            new_df, flag = self._judge_next(pdf, cnt, df)
        if new_df is None:
            return df.to_markdown().replace(" ",'')
        else:
            return new_df.to_markdown().replace(" ",'')
        
    def lazy_parse(self, blob: Blob) -> list[Document]:
        """Lazily parse the blob."""
        import pdfplumber

        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)  # open document
            res = []
            global ps
            ps = []
            for p in range(len(doc.pages)):
                # print("被跳过的页数:",ps)
                if p in ps:
                    continue
                # print("当前页面是:",p)
                page = doc.pages[p]
                min_x, min_y, max_x, max_y = page.bbox[0], page.bbox[1], page.bbox[2], page.bbox[3]
                page_text = ""
                if page.find_tables() != []:
                    tables = page.find_tables()
                    # print(tables)
                    tables_pos = [tables[i].bbox for i in range(len(tables))]
                    # print(tables_pos)
                    for idx, pos in enumerate(tables_pos):
                        x1,y1,x2,y2 = pos

                        if idx == 0:
                            pre_text = page.within_bbox((min_x,min_y,x2,y1)).extract_text()
                        else:
                            pre_text = page.within_bbox((x1_,y2_,x2,y1)).extract_text()
                        page_text += pre_text

                        # 判断是否是跨页表格，需要递归
                        if max_y - y2 <= 100:
                            table_ = self._continue_table(doc, p, page.within_bbox((x1,y1,max_x, max_y)).extract_table())
                            page_text += table_
                        else:
                            table_ = self._table2md(page.within_bbox((x1,y1,x2,y2)).extract_table()).to_markdown().replace(" ",'')
                            page_text += table_
                        
                        x1_, y1_, x2_, y2_ = pos
                            
                    # 有页面尾，一定要加
                    pos_text = page.within_bbox((tables_pos[-1][0],tables_pos[-1][3],max_x,max_y)).extract_text()
                    page_text += pos_text
                    
                else:
                    text = page.extract_text()
                    page_text += text

                res.append(
                    Document(
                    page_content=page_text
                    + "\n"
                    + self._extract_images_from_page(page),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.page_number - 1,
                            "total_pages": len(doc.pages),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                )
        return iter(res)  

    def _process_page_content(self, page: pdfplumber.page.Page) -> str:
        """Process the page content based on dedupe."""
        if self.dedupe:
            return page.dedupe_chars().extract_text(**self.text_kwargs)
        else:
            return page.extract_text(**self.text_kwargs)

    def _extract_images_from_page(self, page: pdfplumber.page.Page) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ""

        images = []
        for img in page.images:
            if img["stream"]["Filter"].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(
                    np.frombuffer(img["stream"].get_data(), dtype=np.uint8).reshape(
                        img["stream"]["Height"], img["stream"]["Width"], -1
                    )
                )
            elif img["stream"]["Filter"].name in _PDF_FILTER_WITH_LOSS:
                images.append(img["stream"].get_data())
            else:
                warnings.warn("Unknown PDF Filter!")

        return extract_from_images_with_rapidocr(images)


class BaseBlobParser(ABC):
    """Abstract interface for blob parsers.

    A blob parser provides a way to parse raw data stored in a blob into one
    or more documents.

    The parser can be composed with blob loaders, making it easy to reuse
    a parser independent of how the blob was originally loaded.
    """

    @abstractmethod
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazy parsing interface.

        Subclasses are required to implement this method.

        Args:
            blob: Blob instance

        Returns:
            Generator of documents
        """

    def parse(self, blob: Blob) -> List[Document]:
        """Eagerly parse the blob into a document or documents.

        This is a convenience method for interactive development environment.

        Production applications should favor the lazy_parse method instead.

        Subclasses should generally not over-ride this parse method.

        Args:
            blob: Blob instance

        Returns:
            List of documents
        """
        return list(self.lazy_parse(blob))
        # return self.lazy_parse(blob)