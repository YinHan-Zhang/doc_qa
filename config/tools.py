tools = [
    {
        "type": "function",
        "function": {
            "name": "query_train_info",
            "description": "根据用户提供的信息，查询对应的车次",
            "parameters": {
                "type": "object",
                "properties": {
                    "departure": {
                        "type": "string",
                        "description": "出发城市或车站",
                    },
                    "destination": {
                        "type": "string",
                        "description": "目的地城市或车站",
                    },
                    "date": {
                        "type": "string",
                        "description": "要查询的车次日期",
                    },
                },
                "required": ["departure", "destination", "date"],
            },
            "name": "query_temperture_info",
            "description": "根据用户提供的信息，查询当天的天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "temperature": {
                        "type": "string",
                        "description": "当前温度",
                    },
                    "wind": {
                        "type": "string",
                        "description": "当前风力",
                    },
                    "rainy": {
                        "type": "string",
                        "description": "今天天气预报，是否会下雨？",
                    },
                },
                "required": ["temperature", "wind", "rainy"],
            },
            "web_search":{
                "enable" : False,
                "search_query" : "注意日期，保证实时性。"
            }
        }
    }
]