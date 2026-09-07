import logging
import json
import os
import time
import re
from nltk.corpus import wordnet
from common.file_utils import get_project_base_directory


# 强制 NLTK 一次性同步加载完整词网（WordNet）语料库，
# 防止并发任务多线程执行时因触发延迟加载（lazy-loading）而发生竞态冲突
try:
    wordnet.ensure_loaded()
except Exception:
    logging.warning("Fail to load wordnet.ensure_loaded()")


class Dealer:
    """同义词字典管理与查询服务 —— 同义词发牌官。"""

    def __init__(self, redis=None):
        """初始化同义词管理器，加载本地静态词典并配置可选的 Redis 缓存连接 —— 同义词管理器初始化工。

        传入参数：
            redis (object | None): 可选的 Redis 客户端连接实例，若不提供则禁用实时动态同义词更新，例如：
                # 传入真实的 Redis 连接实例
                redis = <redis.client.Redis object>
                # 或未配置 Redis 时传入
                redis = None

        返回值：
            None
        """
        # 查询计数器，设为极大的初始值（100,000,000）以确保首次启动执行 load() 时必然满足 > 100 次的更新阈值
        self.lookup_num = 100000000
        # 上次从 Redis 同步词典的时间戳，初始减去 1,000,000 秒以规避 3600 秒（1小时）的冷却检查，确保首次必能加载
        self.load_tm = time.time() - 1000000
        # 内存中存储的同义词字典，格式为 {词: [同义词列表]} 或 {词: 单个同义词}
        self.dictionary = None
        # 定位项目内置的基础同义词词典文件：<项目根目录>/rag/res/synonym.json
        path = os.path.join(get_project_base_directory(), "rag/res", "synonym.json")
        try:
            with open(path, "r") as f:
                self.dictionary = json.load(f)

            # 将词典所有键统一转换为小写，保证后续查词时不区分英文大小写
            self.dictionary = {(k.lower() if isinstance(k, str) else k): v for k, v in self.dictionary.items()}
        except Exception:
            logging.warning("Missing synonym.json")
            # 若文件不存在或读取损坏，初始化为空字典保证程序不崩溃
            self.dictionary = {}

        # 校验 Redis 连接：若无 Redis 则警告无法使用动态更新功能
        if not redis:
            logging.warning("Realtime synonym is disabled, since no redis connection.")
        # 校验本地词典：若字典为空则记录告警
        if not len(self.dictionary.keys()):
            logging.warning("Fail to load synonym")

        self.redis = redis
        # 尝试触发首次从 Redis 加载热更新词典
        self.load()

    def load(self):
        """检查刷新间隔与查询频次，按需从 Redis 中热加载最新同义词字典 —— 同义词热更新检查工。

        传入参数：
            无（直接读取实例属性 self.redis、self.lookup_num 与 self.load_tm）

        返回值：
            None
        """
        # 未配置 Redis 连接时直接退出
        if not self.redis:
            return

        # 节流条件 1：累计查词次数未达到 100 次时跳过刷新，减少对 Redis 的无效高频访问
        if self.lookup_num < 100:
            return
        tm = time.time()
        # 节流条件 2：距离上次加载时间不足 3600 秒（1 小时）时跳过刷新，控制加载周期
        if tm - self.load_tm < 3600:
            return

        # 重置加载时间戳为当前时间，重置查词计数器为 0
        self.load_tm = time.time()
        self.lookup_num = 0
        # 从 Redis 获取键为 "kevin_synonyms" 的最新同义词 JSON 文本
        d = self.redis.get("kevin_synonyms")
        if not d:
            return
        try:
            # 解析 JSON 字符串并覆盖内存中的同义词字典
            d = json.loads(d)
            self.dictionary = d
        except Exception as e:
            logging.error("Fail to load synonym!" + str(e))

    def lookup(self, tk, topn=8):
        """根据输入词元查询其候选同义词列表，优先匹配自定义词典，英文未命中时降级走 WordNet —— 同义词查词工。

        传入参数：
            tk (str): 待查询同义词的目标词元字符串，例如：
                "happy" 或 "自然语言处理"
            topn (int): 最多返回的同义词数量上限（默认 8），例如：
                8

        返回值：
            list[str]: 匹配到的同义词字符串列表（长度不超过 topn），若未找到则返回空列表，例如：
                ["glad", "cheerful", "contented"]
                # 或者未找到匹配项时：
                []
        """
        # 入参有效性检查：词元为空或非字符串类型时直接返回空列表
        if not tk or not isinstance(tk, str):
            return []

        # 步骤 1：查询自定义词典（优先）
        # 累计查询计数，并在满足条件时触发 Redis 异步更新检查
        self.lookup_num += 1
        self.load()
        # 将词元两端空白去除，并将中间连续的空白/制表符替换为单空格
        key = re.sub(r"[ \t]+", " ", tk.strip())
        res = self.dictionary.get(key, [])
        # 统一转为列表结构（兼容词典中值为单个字符串的情况）
        if isinstance(res, str):
            res = [res]
        # 自定义词典命中，截取前 topn 个同义词直接返回
        if res:
            return res[:topn]

        # 步骤 2：降级兜底 —— 若自定义词典未命中且词元为纯英文字母，降级到 WordNet 词网中检索
        if re.fullmatch(r"[a-z]+", tk):
            # 获取该英文单词的所有同义词集（synsets），提取名称并把下划线替换为空格（如 "look_after" -> "look after"）
            wn_set = {re.sub("_", " ", syn.name().split(".")[0]) for syn in wordnet.synsets(tk)}
            # 排除目标词元自身，避免将原词当作同义词返回
            wn_set.discard(tk)
            # 过滤掉空字符串，构成最终的同义词候选列表
            wn_res = [t for t in wn_set if t]
            return wn_res[:topn]

        # 步骤 3：两路均未查到同义词，返回空列表
        return []


if __name__ == "__main__":
    # 本地测试入口：初始化同义词发牌官实例并打印加载到的词典内容
    dl = Dealer()
    print(dl.dictionary)
