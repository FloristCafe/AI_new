import spacy
import pandas as pd

# 1. 挂载引擎
nlp = spacy.load("zh_core_web_sm")

# 2. 李倩的原始脏语料
raw_text = "2026年，字节跳动的创始人张一鸣在北京宣布了一项关于大模型的新计划。随后，百度的李彦宏也发表了讲话。"

doc = nlp(raw_text)

# 3. 准备一个空的 Python 列表，用来装载我们要抽取的“实体列”
entity_data = []

# 4. 暴力提取并结构化
for ent in doc.ents:
    # 只提取我们关心的人名(PERSON)和组织(ORG)
    if ent.label_ in ["PERSON", "ORG"]:
        # 把数据组装成字典（一种极易向表格转换的数据结构）
        entity_data.append({
            "entity": ent.text,
            "label": ent.label_,
            "start_offset": ent.start_char,
            "end_offset": ent.end_char
        })

# 5. 核心降维打击：将散乱的列表瞬间实例化为 Pandas 二维矩阵
df = pd.DataFrame(entity_data)

print("========== 清洗后的高维实体关系表 ==========")
print(df)

# 6. 工业级输出：把这个矩阵砸到硬盘上，这就是你要交付给李倩的成果雏形
df.to_csv("clean_entities.csv", index=False, encoding="utf-8-sig")
print("\n[系统提示] 数据已落盘为 clean_entities.csv")