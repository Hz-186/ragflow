import json
import logging
import math
import os
import re
import unicodedata

import numpy as np

from common.file_utils import get_project_base_directory
from rag.nlp import rag_tokenizer



# 分词器词典（Vocabulary）不可能收录全世界所有的专有名词和技术术语（
# 如 kubernetes、superconductivity、alpha-helix 等）。对于这些
# 词典外的生词（OOV, Out-Of-Vocabulary），传统的处理要么直接置 0，
# 要么给一个固定值。如果直接置 0 或给相同默认值，就会导致两个生词被平
# 等对待（例如短生词 was、the 和罕见的超长专业术语 dichlorodiphenyltrichloroethane 权重一样）。
def _alphabetic_oov_frequency(term):
    """估算由拉丁字母、希腊字母或西里尔字母构成的未登录词（OOV）先验词频 —— 西文字母生词词频估算器。

    传入参数：
        term (str): 待评估的西文单词或词组字符串，例如：

            "kubernetes"

            或

            "alpha-helix"

    返回值：
        int | None: 估算出的词频先验数值；若包含非允许字符则返回 None，例如：
            300   # 极短生词（<=3 字符）的基准频率
            75    # 较长生词估算出的较低频率（赋予更高重要性）
            None  # 包含非法字符或非西文字母
    """
    # 统计词中有效西文字母的个数
    letter_count = 0
    for char in term:
        # 针对 ASCII 字符分支
        if char.isascii():
            if char.isalpha():
                letter_count += 1
            # 允许西文词组中包含空格、点、连字符（如 "st. petersburg" 或 "co-op"），其余字符视为非法
            elif char not in " .-":
                return None
            continue

        # 针对非 ASCII 字符分支，通过 Unicode 字符名检测字母书写系统（Script）
        script_name = unicodedata.name(char, "").split(" ", 1)[0]
        # 仅放行拉丁字母（如带音标字符）、希腊字母或西里尔字母
        if char.isalpha() and script_name in {"LATIN", "GREEK", "CYRILLIC"}:
            letter_count += 1
        elif char not in " .-":
            return None

    # 如果整个词中没有包含任何有效字母，直接判定为无效
    if not letter_count:
        return None

    # 词频衰减计算（信息量越大的长生词，赋予更低的词频，从而在检索时获得更高的 IDF 权重）：
    # 1. 长度不超过 3 的短词保留历史基准词频 300；
    # 2. 超过 3 个字母后，每多 2 个字母，词频减半一次（2^exponent）；
    # 3. 设置保底词频下限为 10，避免词频无底线衰减导致未知生词权重无限膨胀超出已知词汇。
    exponent = max(0, letter_count - 3) / 2
    return max(10, round(300 / (2**exponent)))
    # 短单词（ ≤ 3 字符，如 the, car）：保持基准高词频 300；
    # 超过 3 个字母后，每多 2 个字母，词频减半（如 5 字母词频约为 150，7 字母约为 75，9 字母约为 37……）；
    # 保底下限：设为 10。


class Dealer:
    """词权重计算与查询分词处理器 —— 负责文本预分词、碎片词拼合、命名实体识别及词权重（IDF/NER/POS）综合计算。"""

    def __init__(self):
        """初始化词权重处理器，加载停用词表、实体类型字典与词频统计数据 —— 词典与资源加载器。

        传入参数：
            无显式入参（仅类实例 self）。

        返回值：
            None: 构造函数无返回值，直接完成实例属性初始化。
        """
        # 初始化搜索预置停用词表（过滤无检索价值的代词、介词、助词与疑问词）
        self.stop_words = set(
            [
                "请问",
                "您",
                "你",
                "我",
                "他",
                "是",
                "的",
                "就",
                "有",
                "于",
                "及",
                "即",
                "在",
                "为",
                "最",
                "有",
                "从",
                "以",
                "了",
                "将",
                "与",
                "吗",
                "吧",
                "中",
                "#",
                "什么",
                "怎么",
                "哪个",
                "哪些",
                "啥",
                "相关",
            ]
        )

        # 内部辅助函数：读取以制表符（\t）分隔的词频词典文件
        def load_dict(fnm):
            res = {}
            with open(fnm, "r", encoding="utf-8") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    arr = line.replace("\n", "").split("\t")
                    # 第一列为词，第二列为频次；若无频次列则默认为 0
                    if len(arr) < 2:
                        res[arr[0]] = 0
                    else:
                        res[arr[0]] = int(arr[1])

            # 统计全部词频总和
            c = 0
            for _, v in res.items():
                c += v
            # 若所有词频总和为 0，退化返回纯词集合 set
            if c == 0:
                return set(res.keys())
            return res


        # 确定资源文件目录路径（位于 rag/res 目录下）
        fnm = os.path.join(get_project_base_directory(), "rag/res")
        self.ne, self.df = {}, {}
        # 1. 尝试加载命名实体识别字典（ner.json），用于实体加权（如公司、学校、地名等）
        # 命名实体识别（NER）知识库，记录了各类专有名词的类型（如企业 corp、高校 sch、地名 loca、股 stock、功能词 func、敏感词 toxic）
        try:
            with open(os.path.join(fnm, "ner.json"), "r", encoding="utf-8") as f:
                self.ne = json.load(f)
        except Exception:
            logging.warning("Load ner.json FAIL!")


        # 2. 尝试加载全局词频/文档频率词典（term.freq），用于计算逆文档频率 IDF
        freq_path = os.path.join(fnm, "term.freq")
        try:
            self.df = load_dict(freq_path)
        except FileNotFoundError:
            # 允许可选的词频字典缺失，走默认未登录词先验回退逻辑
            pass
        except (OSError, ValueError):
            logging.warning("Load term.freq FAIL!", exc_info=True)




    def pretoken(self, txt, num=False, stpwd=True):
        """对原始文本进行基础分词、标点过滤与停用词清洗 —— 预分词与噪声词过滤器。

        传入参数：
            txt (str): 待分词的原始文本字符串，例如：
                "请问2024年RAGFlow在GitHub上有多少stars？"
            num (bool): 是否保留单字独立数字（默认 False，即过滤像 "1", "2" 这样的单个阿拉伯数字）。
            stpwd (bool): 是否过滤停用词（默认 True，即过滤停用词表中的无意义词）。

        返回值：
            list[str]: 初步清洗后的词元列表，例如：
                ["2024", "年", "RAGFlow", "GitHub", "stars"]
        """
        # 常见各类标点符号、特殊符号的正则集合
        patt = [r"[~—\t @#%!<>,\.\?\":;'\{\}\[\]_=\(\)\|，。？》•●○↓《；‘’：“”【¥ 】…￥！、·（）×`&\\/「」\\]"]
        # 保留拓展替换规则列表（当前未启用，保留结构以备扩展）
        rewt = []
        for p, r in rewt:
            txt = re.sub(p, r, txt)

        res = []
        # 调用系统分词器切词后，按空格遍历每个候选词元
        for t in rag_tokenizer.tokenize(txt).split():
            tk = t
            # 过滤逻辑：命中止用词（且开启了 stpwd）或单字阿拉伯数字（且 num 为 False 时跳过）
            # 理由：单字数字与纯语气词往往引入巨大的召回噪声
            if (stpwd and tk in self.stop_words) or (re.match(r"[0-9]$", tk) and not num):
                continue
            # 若词元匹配了特殊标点正则，将其标记为占位符 "#"
            for p in patt:
                if re.match(p, t):
                    tk = "#"
                    break
            # 剔除无效占位符与空词，收录有效词元
            if tk != "#" and tk:
                res.append(tk)
        return res



    def token_merge(self, tks):
        """将过度切分的单字修饰词或连续短字符碎片合并为合理词元 —— 碎片词元合并器。

        传入参数：
            tks (list[str]): 预分词产生的词元列表，可能包含被切碎的单字或短字母，例如：
                ["多", "工位", "机", "床"]

        返回值：
            list[str]: 碎片合并重组后的词元列表，例如：
                ["多 工位", "机 床"]
        """
        # 判断词元是否为单字符或 1~2 位的短字母/短数字碎片
        def one_term(t):
            return len(t) == 1 or re.match(r"[0-9a-z]{1,2}$", t)

        res, i = [], 0

        # 若连续碎片长度在 2 到 4 个之间（如“机”+“床” 或 “大”+“模”+“型”），用空格拼成复合词组；
        # 若连续单字碎片超过 5 个（说明原本就是一串零散文字，而非紧凑专有名词），只保守合并前 2
        # 个，防止把一大段话错误焊合成一个长串。

        while i < len(tks):
            j = i
            # 特殊前缀合并：若首词是单字修饰语（如“多”），且紧邻词是多字且非纯西文（如“工位”），
            # 则合并前两个词（例如 "多" + "工位" -> "多 工位"），指针跳跃 2 步
            if i == 0 and one_term(tks[i]) and len(tks) > 1 and (len(tks[i + 1]) > 1 and not re.match(r"[0-9a-zA-Z]", tks[i + 1])):  # 多 工位
                res.append(" ".join(tks[0:2]))
                i = 2
                continue

            # 探测向后连续的单字符/短碎片片段（忽略停用词）
            while j < len(tks) and tks[j] and tks[j] not in self.stop_words and one_term(tks[j]):
                j += 1
            # 若发现连续多个短碎片（> 1）
            if j - i > 1:
                # 若碎片数适中（2~4个），整体拼接为一个完整词组
                if j - i < 5:
                    res.append(" ".join(tks[i:j]))
                    i = j
                # 若碎片过多（>=5个），保守只合并相邻的前两个，防止不相关长文本被错误串接成一大块
                else:
                    res.append(" ".join(tks[i : i + 2]))
                    i = i + 2
            else:
                # 普通完整词直接收录，指针前进 1 步
                if len(tks[i]) > 0:
                    res.append(tks[i])
                i += 1

        return [t for t in res if t]

    def ner(self, t):
        """查询指定词元在命名实体字典中的实体类别标签 —— 命名实体标签查询工。

        传入参数：
            t (str): 待查询的词元字符串，例如：
                "清华大学"

        返回值：
            str: 该词对应的实体类别标识代码；未命中或词典不存在时返回空字符串，例如：
                "sch"   # 学校
                "corp"  # 企业
                ""      # 未收录实体
        """
        # 若命名实体字典未成功加载，直接返回空字符串
        if not self.ne:
            return ""
        # 从词典中获取对应的类别标签
        res = self.ne.get(t, "")
        if res:
            return res

    def split(self, txt):
        """按空白符切分文本并将相邻的西文连续单词重组为复合短语 —— 英文连续词组合并切分工。

        传入参数：
            txt (str): 待切分的文本字符串，例如：
                "learn deep learning and ragflow today"

        返回值：
            list[str]: 切分并合并西文短语后的词列表，例如：
                ["learn deep learning", "and", "ragflow today"]
        """
        tks = []
        # 将连续的空格/制表符替换为单空格，并按空格拆分成初始单词列表
        for t in re.sub(r"[ \t]+", " ", txt).split():
            # 判断是否与前一个词合并：
            # 条件：前一个词和当前词都以英文字母结尾，且两者都不是功能性连接词（func），
            # 则判定它们属于同一复合短语，以空格拼接到前一个词的末尾
            if tks and re.match(r".*[a-zA-Z]$", tks[-1]) and re.match(r".*[a-zA-Z]$", t) and tks and self.ne.get(t, "") != "func" and self.ne.get(tks[-1], "") != "func":
                tks[-1] = tks[-1] + " " + t
            else:
                tks.append(t)
        return tks

    def weights(self, tks, preprocess=True):
        """融合实体类型、词性标注、词频与文档频率计算词元的归一化重要性权重 —— 检索词重要性权重计算器。

        传入参数：
            tks (list[str]): 待计算权重的词元或输入文本列表，例如：
                ["RAGFlow", "知识库", "教程"]
            preprocess (bool): 是否对输入词元执行预分词清洗与碎片重组合并（默认 True）。

        返回值：
            list[tuple[str, float]]: 包含词元与对应归一化权重（各权重之和为 1.0）的二元组列表，例如：
                [
                    ("RAGFlow", 0.45),
                    ("知识库", 0.35),
                    ("教程", 0.20),
                ]
        """

        # 预编译正则模式：
        # 1. 纯数字/带标点的多位数值模式（如 "2024", "3.14"）
        num_pattern = re.compile(r"[0-9,.]{2,}$")
        # 2. 1~2 位短英文字母模式（如 "in", "to", "a"）
        short_letter_pattern = re.compile(r"[a-z]{1,2}$")
        # 3. 带空格或连字符的数字模式（如 "12 - 34"）
        num_space_pattern = re.compile(r"[0-9. -]{2,}$")

        # 内部评分工：基于命名实体类别计算实体重要性权重系数
        def ner(t):
            # 多位数值赋予 2 倍权重（数值通常包含具体参数或版本，信息量高）
            if num_pattern.match(t):
                return 2
            # 1~2 位的极短英文字符赋予 0.01 极低权重（虚词或碎片，检索意义微弱）
            if short_letter_pattern.match(t):
                return 0.01
            # 未收录实体或无实体字典，默认权重为 1
            if not self.ne or t not in self.ne:
                return 1
            # 针对不同实体类别的倍率加权映射：
            # corp（企业）、loca（地名）、sch（高校）、stock（股票）是检索最核心的实体，给予 3 倍权重；
            # toxic（敏感词/有毒有害词）给予 2 倍权重；
            # func（功能词）、firstnm（姓氏）给予 1 倍权重。
            m = {"toxic": 2, "func": 1, "corp": 3, "loca": 3, "sch": 3, "stock": 3, "firstnm": 1}
            return m[self.ne[t]]

        # 内部评分工：基于词性标注（POS）计算词性权重系数
        def postag(t):
            t = rag_tokenizer.tag(t)
            # 代词（r）、连词（c）、副词（d）通常是修饰连接成分，赋予 0.3 折扣权重
            if t in set(["r", "c", "d"]):
                return 0.3
            # 地名（ns）、机构团体名（nt）是检索核心专有名词，赋予 3 倍权重
            if t in set(["ns", "nt"]):
                return 3
            # 普通名词（n）赋予 2 倍权重
            if t in set(["n"]):
                return 2
            # 包含数字短横线的标识符（如型号、编号）赋予 2 倍权重
            if re.match(r"[0-9-]+", t):
                return 2
            # 其余常规词性（动词、形容词等）默认赋予 1 倍权重
            return 1

        # 内部评分工：获取或估算词元在通用语料中的绝对词频
        def freq(t):
            # 带空格数字给固定基准值 3
            if num_space_pattern.match(t):
                return 3
            # 从分词器词典中查询词频
            s = rag_tokenizer.freq(t)
            # 若词典中无记录（未登录词 OOV）：
            if not s:
                # 尝试通过西文字母规律估算未登录词词频（长生词词频更低）
                oov_frequency = _alphabetic_oov_frequency(t)
                if oov_frequency is not None:
                    return oov_frequency
            if not s:
                s = 0

            # 若仍未查出且词长大于等于 4，尝试细粒度切词：
            # 取各子词词频的最小值并除以 6.0，作为该复合词的词频估计
            if not s and len(t) >= 4:
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    s = np.min([freq(tt) for tt in s]) / 6.0
                else:
                    s = 0

            # 保底词频下限设为 10，避免词频过低导致 IDF 无限放大
            return max(s, 10)

        # 内部评分工：获取或估算词元的文档频率（DF，出现该词的文档总数）
        def df(t):
            # 带空格数字给固定基准值 5
            if num_space_pattern.match(t):
                return 5
            # 若收录在全局 DF 词典中，加 3 进行平滑后返回
            if t in self.df:
                return self.df[t] + 3
            # 西文生词估算
            oov_frequency = _alphabetic_oov_frequency(t)
            if oov_frequency is not None:
                return oov_frequency
            # 长词尝试细粒度切分后取子词最小值除以 6.0，保底 3
            if len(t) >= 4:
                s = [tt for tt in rag_tokenizer.fine_grained_tokenize(t).split() if len(tt) > 1]
                if len(s) > 1:
                    return max(3, np.min([df(tt) for tt in s]) / 6.0)

            # 默认保底平滑文档频率为 3
            return 3

        # 内部计算工：平滑逆文档频率（IDF）计算公式
        # 公式：log10(10 + ((N - s + 0.5) / (s + 0.5)))
        # 词频/文档频次 s 越低，该词信息量越大，IDF 得分越高
        def idf(s, N):
            return math.log10(10 + ((N - s + 0.5) / (s + 0.5)))

        tw = []
        if not preprocess:
            # 模式 A：不执行预处理，直接对传入的词元列表计算权重
            # idf1：基于语料总词数 10,000,000 的词频 IDF
            idf1 = np.array([idf(freq(t), 10000000) for t in tks])
            # idf2：基于语料总文档数 1,000,000,000 的文档频率 IDF
            idf2 = np.array([idf(df(t), 1000000000) for t in tks])
            # 综合打分：30% 词频 IDF + 70% 文档频率 IDF，再乘上实体类别加权与词性加权
            wts = (0.3 * idf1 + 0.7 * idf2) * np.array([ner(t) * postag(t) for t in tks])
            wts = [s for s in wts]
            tw = list(zip(tks, wts))
        else:
            # 模式 B：对每个输入词元先清洗切词（pretoken）再进行碎片重组合并（token_merge）
            for tk in tks:
                tt = self.token_merge(self.pretoken(tk, True))
                # 对重组后的子词元分别计算两路 IDF 及实体、词性特征
                idf1 = np.array([idf(freq(t), 10000000) for t in tt])
                idf2 = np.array([idf(df(t), 1000000000) for t in tt])
                wts = (0.3 * idf1 + 0.7 * idf2) * np.array([ner(t) * postag(t) for t in tt])
                wts = [s for s in wts]
                tw.extend(zip(tt, wts))

        # 权重归一化：计算总权重和 S，将每个词元的得分除以 S，使所有词元权重之和为 1.0
        S = np.sum([s for _, s in tw])
        return [(t, s / S) for t, s in tw]
