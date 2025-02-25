import json

data = json.load(open("data.json"))

list = ""
for item in data:
    message = {
        "messages": [
            {
                "role": "system",
                "content": "你是一个电视机上的控制机器人，你的任务是根据用户的指令来控制电视机"
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
with open("tv-data.jsonl", "w",encoding="utf-8") as f:
    f.write(list)
    f.close()
