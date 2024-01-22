tools = [
    {
        "type": "function",
        "function": {
            "name": "query_train_info",
            "description": "�����û��ṩ����Ϣ����ѯ��Ӧ�ĳ���",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "�������л�վ",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Ŀ�ĵس��л�վ",
                    },
                    "date": {
                        "type": "string",
                        "description": "Ҫ��ѯ�ĳ�������",
                    },
                },
                "required": ["departure", "destination", "date"],
            },
            "name": "query_temperture_info",
            "description": "�����û��ṩ����Ϣ����ѯ������������",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "string",
                        "description": "��ǰ�¶�",
                    },
                    "wind": {
                        "type": "string",
                        "description": "��ǰ����",
                    },
                    "rainy": {
                        "type": "string",
                        "description": "��������Ԥ�����Ƿ�����ꣿ",
                    },
                },
                "required": ["temperature", "wind", "rainy"],
            },
            "web_search":{
                "enable" : False,
                "search_query" : "ע�����ڣ���֤ʵʱ�ԡ�"
            }
        }
    }
]