import json
import random
from datetime import datetime, timedelta
from faker import Faker
import re

fake = Faker('zh_CN')

# ====== 参数池配置 ======
USERNAME_POOL = [
    lambda: fake.name(),
    lambda: fake.name_nonbinary(),
    lambda: random.choice(["先生", "女士", "博士", "教授"]) + fake.last_name(),
    lambda: f"{fake.last_name()}-{fake.first_name()}",
    lambda: f"{fake.user_name()}_{random.randint(100, 999)}"
]

BOOKID_POOL = [
    lambda: str(random.randint(100, 99999999)),
    lambda: f"{random.choice('ABCDEFGH')}{random.randint(1000, 9999)}",
    lambda: f"{random.choice(['CX-', 'MU-', 'CA_'])}{random.randint(100, 999)}",
    lambda: ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789', k=6))
]

# 添加城市池
CITY_POOL = [
    "北京", "上海", "广州", "深圳", "成都", "杭州", "重庆", "西安",
    "武汉", "南京", "长沙", "青岛", "厦门", "昆明", "天津", "郑州"
]

VERB_MAP = {
    "detail": ["查询", "查看", "显示", "找一下", "查查", "看看", "帮我看看"],
    "cancel": ["取消", "撤销", "删除", "解除", "终止", "作废"],
    "change": ["修改", "改签", "调整", "变更", "更改", "重新安排"],
    "create": ["订", "预订", "预定", "定", "买", "购买", "订购"]
}


def add_noise(text):
    """添加随机噪声，模拟真实用户输入"""
    noise_types = [
        lambda t: ' '.join(t),
        lambda t: t + random.choice(['。', '！', '～', '…']),
        lambda t: t + random.choice(['啊', '呢', '吧', '哦', '呀']),
        lambda t: t + random.choice(['😊', '👌', '🙏', '😄', '👍']),
        lambda t: duplicate_random_char(t)
    ]

    selected_noise = random.sample(noise_types, random.randint(1, 2))
    for noise_func in selected_noise:
        text = noise_func(text)
    return text


def duplicate_random_char(text):
    """重复文本中的随机字符"""
    if not text:
        return text

    pos = random.randint(0, len(text) - 1)
    char = text[pos]
    return text[:pos] + char * 2 + text[pos + 1:]


def generate_date(days_range=(7, 90)):
    """生成未来日期"""
    future_date = datetime.now() + timedelta(days=random.randint(*days_range))
    return future_date.strftime("%Y-%m-%d %H:%M:%S")


def generate_case(action):
    """生成单个数据样例"""
    username = random.choice(USERNAME_POOL)()
    book_id = random.choice(BOOKID_POOL)()
    greeting = random.choice(["你好", "您好", "Hi", "请问", "麻烦", "幫我"])
    identity = random.choice(["用户名", "会员名", "账户", "客户号"])
    verb = random.choice(VERB_MAP[action])

    # 为订票准备始发地和目的地
    source = random.choice(CITY_POOL)
    target = random.choice([city for city in CITY_POOL if city != source])

    templates = {
        "detail": [
            lambda: f"{greeting}，我的{identity}是{username}，预订编号{book_id}，{verb}航班信息",
            lambda: f"{verb}{greeting}的订单：{username}，编号{book_id}",
            lambda: f"{greeting}，{identity}为{username}，{verb}预订{book_id}的详情"
        ],
        "cancel": [
            lambda: f"{greeting}需要{verb}{book_id}的预订，{identity}是{username}",
            lambda: f"{verb}{username}的{book_id}订单",
            lambda: f"{greeting}{identity}{username}请求{verb}编号{book_id}"
        ],
        "change": [
            lambda: f"{greeting}想把{book_id}改到{date}，{identity}名{username}",
            lambda: f"{verb}{username}的{book_id}至{date}",
            lambda: f"{greeting}{identity}{username}要求{verb}时间到{date}"
        ],
        "create": [
            lambda: f"{greeting}，{identity}是{username}，想{verb}一张从{source}到{target}的机票，{date}出发",
            lambda: f"{verb}机票，{source}飞{target}，{date}，{identity}{username}",
            lambda: f"{greeting}，我是{username}，需要{verb}{date}从{source}到{target}的航班"
        ]
    }

    # 处理需要日期的操作
    date = None
    if action in ["change", "create"]:
        date = generate_date()

    # 生成指令文本
    template_func = random.choice(templates[action])
    instruction = template_func() if action not in ["change", "create"] else template_func().format(date=date)

    # 可能添加噪声
    if random.random() < 0.3:
        instruction = add_noise(instruction)

    # 准备输出
    output = {
        "action": action,
        "params": {
            "username": username,
        }
    }

    # 根据不同action添加参数
    if action in ["detail", "cancel", "change"]:
        output["params"]["bookId"] = book_id
    if action in ["change"]:
        output["params"]["date"] = date
    if action == "create":
        output["params"].update({
            "source": source,
            "target": target,
            "date": date
        })

    return {
        "instruction": instruction,
        "input": "",
        "output": json.dumps(output, ensure_ascii=False)
    }


def main():
    """主程序入口"""
    dataset = []
    # 添加create动作的数量分配
    action_dist = {
        "detail": 400,
        "cancel": 300,
        "change": 300,
        "create": 400  # 添加订票数量
    }

    # 生成数据集
    for action, count in action_dist.items():
        for _ in range(count):
            dataset.append(generate_case(action))

    # 随机打乱数据集
    random.shuffle(dataset)

    # 保存到文件
    output_file = 'flight_dataset.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"已生成 {len(dataset)} 条数据，保存至 {output_file}")

    # 打印几个样例
    print("\n示例数据:")
    for i in range(4):  # 增加到4个样例，确保能看到新增的create类型
        print(f"\n样例 {i + 1}:")
        print(json.dumps(dataset[i], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()