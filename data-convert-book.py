import json

data = json.load(open("flight_dataset.json"))

list = ""
for item in data:
    message = {
        "messages": [
            {
                "role": "system",
                "content": "你是一个机票的助手，你的任务是根据用户的指令来查询、取消、更改机票日期信息"
            },
            {
                "role": "user",
                "content": item["instruction"]
            },
            {
                "role": "assistant",
                "content": item["output"]
            }
        ]
    }
    list += json.dumps(message, ensure_ascii=False) + "\n"

print(list)
with open("tv-book.jsonl", "w",encoding="utf-8") as f:
    f.write(list)
    f.close()
