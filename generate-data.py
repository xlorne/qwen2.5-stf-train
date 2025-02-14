import json
import random
from itertools import product

# 定义基础数据
artists = {
    "周杰伦": ["周杰伦", "杰伦", "Jay"],
    "张学友": ["张学友", "学友", "歌神"],
    "林俊杰": ["林俊杰", "JJ", "俊杰"],
    "五月天": ["五月天", "阿信", "玛莎"],
    "陈奕迅": ["陈奕迅", "Eason", "迅仔"],
    "Beyond": ["Beyond", "beyond", "黄家驹"],
    "李宗盛": ["李宗盛", "宗盛", "老李"],
    "孙燕姿": ["孙燕姿", "燕姿", "孙老师"],
    "张国荣": ["张国荣", "哥哥", "Leslie"],
    "王菲": ["王菲", "天后", "菲姐"],
    "邓丽君": ["邓丽君", "丽君", "邓老师"],
    "刘德华": ["刘德华", "华仔", "华哥"],
    "谭咏麟": ["谭咏麟", "麟仔", "阿伦"],
    "黎明": ["黎明", "Leon", "黎明哥"],
    "莫文蔚": ["莫文蔚", "文蔚", "莫莫"],
    "张惠妹": ["张惠妹", "阿妹", "A-Mei"],
    "郭富城": ["郭富城", "城城", "舞王"],
    "李克勤": ["李克勤", "克勤", "勤哥"]
}

actors = {
    "周星驰": ["周星驰", "星爷", "喜剧之王"],
    "成龙": ["成龙", "大哥", "龙哥"],
    "李连杰": ["李连杰", "李连杰", "武打巨星"],
    "章子怡": ["章子怡", "章子怡", "国际章"],
    "周润发": ["周润发", "发哥", "润发"],
    "梁朝伟": ["梁朝伟", "朝伟", "伟仔"],
    "刘德华": ["刘德华", "华仔", "华哥"],
    "张曼玉": ["张曼玉", "曼玉", "玉姐"],
    "周迅": ["周迅", "迅姐", "迅迅"],
    "巩俐": ["巩俐", "巩俐", "俐姐"],
    "王祖贤": ["王祖贤", "祖贤", "贤姐"],
    "张国荣": ["张国荣", "哥哥", "Leslie"],
    "梁家辉": ["梁家辉", "家辉", "辉哥"],
    "郑秀文": ["郑秀文", "秀文", "Sammi"],
    "甄子丹": ["甄子丹", "子丹", "丹爷"],
    "刘青云": ["刘青云", "青云", "云哥"]
}

# 音乐播放模板
music_templates = [
    "播放{name}的歌",
    "我想听{name}的音乐",
    "来点{name}的",
    "放一首{name}的歌曲",
    "给我放{name}的歌",
    "播放{name}的音乐",
    "想听{name}唱的歌",
    "来几首{name}的歌",
    "我要听{name}",
    "{name}的歌放一下",
    "来点{name}的音乐",
    "帮我放{name}的歌",
    "播放一下{name}的歌",
    "{name}的音乐来一首",
    "放{name}的歌给我听"
]

# 电影查询模板
movie_templates = [
    "{name}演过的电影有哪些？",
    "{name}的电影",
    "{name}演过什么电影",
    "给我看看{name}演的电影",
    "查询{name}的电影",
    "{name}主演的电影有哪些",
    "找一下{name}的电影作品",
    "{name}演过哪些电影",
    "列出{name}的电影",
    "帮我查{name}的电影",
    "{name}演了什么电影",
    "搜索{name}的电影作品",
    "{name}参演的电影",
    "显示{name}的电影列表",
    "给我{name}的电影清单"
]

# 音量调节模板 - 直接使用固定的表达方式避免重复
volume_commands = {
    "increase": [
        "调大声音",
        "声音大一点",
        "把音量调高",
        "音量调大声点",
        "声音调高一些",
        "把声音调大",
        "音量增大",
        "声音放大点",
        "调高音量",
        "音量大点声"
    ],
    "decrease": [
        "调小声音",
        "声音小一点",
        "把音量调低",
        "音量调小声点",
        "声音调低一些",
        "把声音调小",
        "音量减小",
        "声音放小点",
        "调低音量",
        "音量小点声"
    ]
}


def generate_unique_dataset(size=1000):
    all_instructions = set()
    dataset = []

    # 1. 生成所有音量调节指令
    for level, commands in volume_commands.items():
        for command in commands:
            if len(dataset) >= size:
                break
            output = {
                "action": "adjust_volume",
                "level": level
            }
            dataset.append({
                "instruction": command,
                "input": "",
                "output": json.dumps(output, ensure_ascii=False)
            })
            all_instructions.add(command)

    # 2. 生成音乐播放指令
    for artist, names in artists.items():
        for name, template in product(names, music_templates):
            if len(dataset) >= size:
                break
            instruction = template.format(name=name)
            if instruction not in all_instructions:
                output = {
                    "action": "play_music",
                    "artist": artist
                }
                dataset.append({
                    "instruction": instruction,
                    "input": "",
                    "output": json.dumps(output, ensure_ascii=False)
                })
                all_instructions.add(instruction)

    # 3. 生成电影查询指令
    for actor, names in actors.items():
        for name, template in product(names, movie_templates):
            if len(dataset) >= size:
                break
            instruction = template.format(name=name)
            if instruction not in all_instructions:
                output = {
                    "action": "find_movies",
                    "actor": actor
                }
                dataset.append({
                    "instruction": instruction,
                    "input": "",
                    "output": json.dumps(output, ensure_ascii=False)
                })
                all_instructions.add(instruction)

    # 如果生成的数据不足，随机组合生成更多
    while len(dataset) < size:
        # 随机选择生成类型
        gen_type = random.choice(["music", "movie"])

        if gen_type == "music":
            artist = random.choice(list(artists.keys()))
            name = random.choice(artists[artist])
            template = random.choice(music_templates)
            instruction = template.format(name=name)
        else:  # movie
            actor = random.choice(list(actors.keys()))
            name = random.choice(actors[actor])
            template = random.choice(movie_templates)
            instruction = template.format(name=name)

        if instruction not in all_instructions:
            output = {
                "action": "play_music" if gen_type == "music" else "find_movies",
                "artist" if gen_type == "music" else "actor": artist if gen_type == "music" else actor
            }
            dataset.append({
                "instruction": instruction,
                "input": "",
                "output": json.dumps(output, ensure_ascii=False)
            })
            all_instructions.add(instruction)

    # 随机打乱数据集顺序
    random.shuffle(dataset)
    return dataset[:size]


# 生成数据集
dataset = generate_unique_dataset(1000)
print(json.dumps(dataset, ensure_ascii=False, indent=2))

outfile = 'train.json'
with open(outfile, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
print(f"Saved dataset to {outfile}")