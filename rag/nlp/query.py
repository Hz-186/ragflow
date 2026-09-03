#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import logging
import json
import re
from collections import defaultdict

from common.query_base import QueryBase
from common.doc_store.doc_store_base import MatchTextExpr
from rag.nlp import rag_tokenizer, term_weight, synonym
from rag.utils.redis_conn import REDIS_CONN


class FulltextQueryer(QueryBase):
    """全文「查询构建器」—— 把自然语言问题翻译成文档引擎能执行的加权查询语句。

    三类职责：
    1. question()：用户问题 → 带权重、带同义词、带词组搭配的全文查询表达式
       （search.py 的 Dealer.search 用它做混合查询的文本半边）；
    2. paragraph()：切片内容 → 「找相似切片」的查询表达式（入库算标签特征用）；
    3. 相似度家族（hybrid_similarity / token_similarity / similarity）：
       在本地算「词重叠」相似度，供重排打分和引用插桩使用。

    内部两员大将：
        self.tw  = term_weight.Dealer()  # 词权重器：IDF × 实体类型 × 词性，给每个词打分
        self.syn = synonym.Dealer()      # 同义词表：优先自定义词典（Redis 里），兜底 WordNet
    """

    def __init__(self):
        """装配查询构建器：准备词权重器、同义词表，并声明「查哪些字段、各字段权重」。

        query_fields 是全文查询的目标字段清单，^ 后是字段权重（默认 1）：
            title_tks^10       # 标题粗分词 —— 命中标题价值是正文的 10 倍
            title_sm_tks^5     # 标题细分词
            important_kwd^30   # 关键词（入库时由 LLM 提取）—— 权重最高
            important_tks^20   # 关键词分词
            question_tks^20    # QA 对的「问题」分词（QA 文档专用字段）
            content_ltks^2     # 正文粗分词（BM25 主力字段）
            content_sm_ltks    # 正文细分词（兜底，权重 1）
        同义词词典优先从 Redis 加载（Redis 不可用就只用本地词典 + WordNet）。
        """
        self.tw = term_weight.Dealer()
        self.syn = synonym.Dealer(redis=REDIS_CONN.REDIS if REDIS_CONN.is_alive() else None)
        self.query_fields = [
            "title_tks^10",
            "title_sm_tks^5",
            "important_kwd^30",
            "important_tks^20",
            "question_tks^20",
            "content_ltks^2",
            "content_sm_ltks",
        ]

    def question(self, txt, tbl="qa", min_match: float = 0.6):
        """把用户问题加工成「带权重的全文查询表达式」—— 问题翻译器（本类的主入口）。

        输入参数的样子：
            txt = "How does naive_merge work?"   # 用户问题原文（中英文都行）
            tbl = "qa"                           # 历史遗留参数，当前实现未使用
            min_match = 0.6                      # 最低命中率：至少多少比例的词条要匹配上。
                                                 # search.py 首查传 0.3（近纯向量搜索时传 0），
                                                 # 空结果重试时传 0.1

        返回值的样子（二元组）：
            (
                MatchTextExpr(
                    fields=["title_tks^10", ...],        # 查哪些字段（见 __init__）
                    matching_text="(merge^1.23 \"blend\"^0.31) (chunk^0.89) \"merge chunk\"^1.78",
                                                         # 查询体：每个词带权重、带同义词，
                                                         # 相邻词还额外组成短语，整体是引擎查询语法
                    topn=100,
                    extra_options={"minimum_should_match": 0.6, "original_query": "原始问题"},
                                                         # 注意：minimum_should_match 只有中文路线才带；
                                                         # 英文路线的 extra_options 里没有它
                ),
                ["merge", "chunk", "blend", ...],        # 关键词列表（含同义词、细粒度子词，
                                                         # 供高亮和后续扩展用）
            )
            (None, [])                                    # 问题分不出任何词时
        """
        original_query = txt  # 保存原文，最后塞进 extra_options 供日志/排查用
        txt = self.add_space_between_eng_zh(txt)  # 中英文交界处补空格，方便后续切分

        # 把 Infinity 的「可转义特殊字符」从查询里清掉。
        # Infinity 的词法分析器（search_lexer.l）定义了这些特殊字符：[\x20()^"'~*?:\\]
        # 它们不转义就出现在查询里的话，会被当成查询语法符号，导致解析报错。
        # 顺带做了三件归一化：小写、全角转半角（strQ2B）、繁体转简体（tradi2simp）
        txt = re.sub(
            r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>*~'\"\\]+",
            " ",
            rag_tokenizer.tradi2simp(rag_tokenizer.strQ2B(txt.lower())),
        ).strip()
        if not rag_tokenizer.tokenize(txt).strip():
            return None, []  # 全是标点/空白，分不出词，没法查
        otxt = txt
        txt = self.rmWWW(txt)  # 去掉疑问语气词（什么/怎么/what/how 等），保留实义内容

        if not self.is_chinese(txt):
            # ===== 英文（非中文）路线 =====
            txt = self.rmWWW(txt)
            tks = rag_tokenizer.tokenize(txt).split()  # 分词（含词干化/词形还原）
            keywords = [t for t in tks if t]
            tks_w = self.tw.weights(tks, preprocess=False)  # 每个词算权重：[(词, 权重), ...]
            # 三轮清洗：去掉会干扰引擎语法的字符（反斜杠/引号/脱字符、开头 +/-、首尾空白）
            tks_w = [(re.sub(r"[ \\\"'^]", "", tk), w) for tk, w in tks_w]
            tks_w = [(re.sub(r"^[\+-]", "", tk), w) for tk, w in tks_w if tk]
            tks_w = [(tk.strip(), w) for tk, w in tks_w if tk.strip()]
            syns = []
            # 为每个词查同义词，拼成「主词 + 同义词」的加权组
            for tk, w in tks_w[:256]:
                # 同义词里的单引号要去掉，否则 Infinity 词法器会报 TokenError
                #（WordNet 会给 "cat" 返回 "cat-o'-nine-tails" 这种带撇号的词）
                syn = [rag_tokenizer.tokenize(s).replace("'", "") for s in self.syn.lookup(tk)]
                keywords.extend(syn)  # 同义词也收进关键词（高亮时能命中）
                syn = ['"{}"^{:.4f}'.format(s, w / 4.0) for s in syn if s.strip()]  # 同义词权重打主词的 1/4
                syns.append(" ".join(syn))

            # 每个词一个括号组：(主词^权重 同义词^权重 ...)；
            # 以 . ^ + ( ) - 开头的词是引擎保留语法，跳过
            q = ["({}^{:.4f}".format(tk, w) + " {})".format(syn) for (tk, w), syn in zip(tks_w, syns) if tk and not re.match(r"[.^+\(\)-]", tk)]
            # 相邻两词再组成「短语查询」，权重取两者较大值的 2 倍 ——
            # 原文里挨着的词组（如 machine learning）理应获得更高匹配分
            for i in range(1, len(tks_w)):
                left, right = tks_w[i - 1][0].strip(), tks_w[i][0].strip()
                if not left or not right:
                    continue
                q.append(
                    '"%s %s"^%.4f'
                    % (
                        tks_w[i - 1][0],
                        tks_w[i][0],
                        max(tks_w[i - 1][1], tks_w[i][1]) * 2,
                    )
                )
            if not q:
                q.append(txt)  # 兜底：所有词都被过滤掉时，用清洗后的原文裸查
            query = " ".join(q)
            # 英文路线不设 minimum_should_match，由引擎默认规则决定命中比例
            return MatchTextExpr(self.query_fields, query, 100, {"original_query": original_query}), keywords

        # ===== 中文路线 =====

        def need_fine_grained_tokenize(tk):
            """判断一个词值不值得再细切：太短（<3 字符）或纯数字/符号/代码词不切。"""
            if len(tk) < 3:
                return False
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                return False
            return True

        txt = self.rmWWW(txt)
        qs, keywords = [], []
        # tw.split：按空白切词组，相邻英文词会粘回一个词组（"New York" 不被拆散），最多 256 组
        for tt in self.tw.split(txt)[:256]:
            if not tt:
                continue
            keywords.append(tt)
            twts = self.tw.weights([tt])  # 词组内部再细分并加权，可能拆成多个子词
            syns = self.syn.lookup(tt)  # 整个词组的同义词
            if syns and len(keywords) < 32:
                keywords.extend(syns)
            logging.debug(json.dumps(twts, ensure_ascii=False))
            tms = []
            # 按权重从高到低处理词组的每个子词
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                # 细粒度子词：把长词再切碎（如「机器学习」→「机器 学习」），碎词也参与匹配
                sm = rag_tokenizer.fine_grained_tokenize(tk).split() if need_fine_grained_tokenize(tk) else []
                # 子词清洗：删标点 → 转义引擎特殊字符 → 丢弃单字符碎屑
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                        "",
                        m,
                    )
                    for m in sm
                ]
                sm = [self.sub_special_char(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]

                # 关键词收集（上限 32 个，够高亮用即可，防膨胀）
                if len(keywords) < 32:
                    keywords.append(re.sub(r"[ \\\"']+", "", tk))
                    keywords.extend(sm)

                # 子词级别的同义词：转义、收进关键词、细粒度化；含空格的多词同义词加引号当短语
                tk_syns = self.syn.lookup(tk)
                tk_syns = [self.sub_special_char(s) for s in tk_syns]
                if len(keywords) < 32:
                    keywords.extend([s for s in tk_syns if s])
                tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
                tk_syns = [f'"{s}"' if s.find(" ") > 0 else s for s in tk_syns]

                if len(keywords) >= 32:
                    break  # 关键词攒够了就停，不再继续拼查询（性能护栏）

                # 拼这一个子词的查询片段：
                tk = self.sub_special_char(tk)
                if tk.find(" ") > 0:
                    tk = '"%s"' % tk  # 多词组合加引号变短语
                if tk_syns:
                    # 同义词用 OR 挂上，权重压低到 0.2（同义匹配只是锦上添花）
                    tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)
                if sm:
                    # 细粒度子词两种形态都挂上：精确短语 + 允许插 2 个词的宽松短语（~2），
                    # 后者权重 0.5，兼顾「切碎也能命中」和「顺序别太乱」
                    tk = f'{tk} OR "%s" OR ("%s"~2)^0.5' % (" ".join(sm), " ".join(sm))
                if tk.strip():
                    tms.append((tk, w))

            # 词组内所有子词片段各自带权重，空格连接（空格即 OR/AND，取决于引擎配置）
            tms = " ".join([f"({t})^{w}" for t, w in tms])

            if len(twts) > 1:
                # 词组被拆成了多个子词时，补一条「整词组的宽松短语」（~2），权重 1.5：
                # 子词在原文里挨得近时给予额外奖励
                tms += ' ("%s"~2)^1.5' % rag_tokenizer.tokenize(tt)

            # 整个词组的同义词拼成 OR 短语
            syns = " OR ".join(['"%s"' % rag_tokenizer.tokenize(self.sub_special_char(s)) for s in syns])
            if syns and tms:
                # 词组整体再包一层：主查询权重 ×5，同义词分支只给 0.7
                tms = f"({tms})^5 OR ({syns})^0.7"

            qs.append(tms)

        if qs:
            # 各词组的查询块之间用 OR 连接：命中任意一组都算数，
            # 命中多少组由 minimum_should_match（即入参 min_match）控制
            query = " OR ".join([f"({t})" for t in qs if t])
            if not query:
                query = otxt  # 兜底：全被过滤空了就用归一化后的原文裸查
            return MatchTextExpr(self.query_fields, query, 100, {"minimum_should_match": min_match, "original_query": original_query}), keywords
        return None, keywords

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
        """「向量余弦 × 词重叠」双路混合相似度 —— 本地打分主力。

        输入参数的样子：
            avec  = [0.1, -0.2, ...]           # 查询（句子/问题）的向量
            bvecs = [[...], [...], ...]        # 一批切片的向量（矩阵）
            atks  = ["机器", "学习", ...]       # 查询的分词列表
            btkss = [["第一章", "介绍"], ...]   # 每个切片的分词列表（加权词袋，可含重复）
            tkweight = 0.3 / vtweight = 0.7    # 两路权重

        返回值的样子（三个与 bvecs 等长的序列）：
            (混合相似度数组, 纯词相似度列表, 纯余弦数组)
            特例：余弦之和为 0（向量不可用）时，混合分直接退化成词相似度。
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        sims = cosine_similarity([avec], bvecs)  # 查询向量 × 切片向量矩阵 → 一行余弦值
        tksim = self.token_similarity(atks, btkss)  # 词重叠相似度（本地算）
        if np.sum(sims[0]) == 0:
            return np.array(tksim), tksim, sims[0]  # 向量路失效，只用词相似度
        return np.array(sims[0]) * vtweight + np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        """把两段文本的「加权词袋」做重叠度比较 —— 词相似度计算器。

        输入参数的样子：
            atks  = ["机器", "学习"]            # 查询的分词（列表或空格分隔字符串均可）
            btkss = [["第一章", "机器"], ...]   # 一批切片的分词

        返回值的样子：
            [0.42, 0.13, ...]   # 查询对每个切片的词相似度（0~1 之间）

        词袋构建规则（to_dict）：单词本身记 0.4×权重，
        相邻两词的拼接词再记 0.6×较大权重 —— 词序信息被部分保留下来。
        """

        def to_dict(tks):
            """分词列表 → {词: 权重} 词袋，单词四成、相邻词对六成。"""
            if isinstance(tks, str):
                tks = tks.split()
            d = defaultdict(int)
            wts = self.tw.weights(tks, preprocess=False)  # 每个词的权重
            for i, (t, c) in enumerate(wts):
                d[t] += c * 0.4  # 单词自身贡献
                if i + 1 < len(wts):
                    _t, _c = wts[i + 1]
                    d[t + _t] += max(c, _c) * 0.6  # 与下一个词的拼接贡献（取较大权重）
            return d

        atks = to_dict(atks)
        btkss = [to_dict(tks) for tks in btkss]
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        """算「查询词袋」在「文档词袋」里的覆盖率 —— 单对文本词相似度。

        输入参数的样子（两种形态都接受）：
            qtwt = {"机器": 0.5, "学习": 0.3}   # 查询词袋 {词: 权重}
            dtwt = "第一章 机器 的 介绍"         # 文档原文字符串（自动现算词袋）
        返回值：0~1 的浮点数。

        算法：把查询词袋里「也出现在文档词袋中」的词权重加起来，除以查询总权重
        —— 即「查询有多大比例的分量被文档接住了」。
        注意只看查询词在不在文档里，不乘文档侧权重（见行内注释的取舍）。
        """
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.tw.weights(self.tw.split(dtwt), preprocess=False)}  # 字符串 → 词袋
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.tw.weights(self.tw.split(qtwt), preprocess=False)}
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v  # 只加查询侧权重，不乘文档侧权重（乘积的区分度反而更差）
        q = 1e-9
        for k, v in qtwt.items():
            q += v  # 查询总权重（分母），1e-9 防止空词袋除零
        return s / q

    def paragraph(self, content_tks: str, keywords: list = [], keywords_topn=30):
        """把「一个切片的内容」拼成找相似用的查询表达式 —— 切片查询生成器。

        用途：入库阶段给切片算标签特征时（search.py 的 tag_content），
        需要拿这个切片的内容去库里搜「邻居」，本方法负责把切片内容拼成查询。

        输入参数的样子：
            content_tks = "第一 章 介绍 合并 切片"   # 已分词的文本（空格分隔，字符串或列表均可）
            keywords = ["合并"]                      # 该切片入库时提取的关键词（会加引号当短语）
            keywords_topn = 30                       # 最多取内容里权重最高的多少个词进查询

        返回值的样子：
            MatchTextExpr(
                fields=["title_tks^10", ...],
                matching_text='"合并" (介绍 OR (讲解)^0.2)^0.31 (切片)^0.28 ...',
                # 结构：原关键词带引号在前；内容词各自带权重挂在括号组外，
                # 有同义词的词用 OR 把同义词组挂进括号内（同义词权重固定 0.2）
                topn=100,
                extra_options={"minimum_should_match": 2,   # 总词数（原关键词+入选内容词）的 1/10
                                                            # 经 round() 取整，至多 3
                               "original_query": "合并"},
            )
        """
        if isinstance(content_tks, str):
            content_tks = [c.strip() for c in content_tks.split() if c.strip()]  # 字符串 → 词列表
        tks_w = self.tw.weights(content_tks, preprocess=False)  # 每个词算权重

        origin_keywords = keywords.copy()  # 备份原关键词，塞进 extra_options 留痕
        keywords = [f'"{k.strip()}"' for k in keywords]  # 关键词加引号当短语查询
        # 内容里权重最高的前 N 个词，逐个带上同义词拼进查询
        for tk, w in sorted(tks_w, key=lambda x: x[1] * -1)[:keywords_topn]:
            tk_syns = self.syn.lookup(tk)
            tk_syns = [self.sub_special_char(s) for s in tk_syns]  # 转义引擎特殊字符
            tk_syns = [rag_tokenizer.fine_grained_tokenize(s) for s in tk_syns if s]
            tk_syns = [f'"{s}"' if s.find(" ") > 0 else s for s in tk_syns]  # 多词同义词加引号
            tk = self.sub_special_char(tk)
            if tk.find(" ") > 0:
                tk = '"%s"' % tk  # 多词组合加引号变短语
            if tk_syns:
                tk = f"({tk} OR (%s)^0.2)" % " ".join(tk_syns)  # 同义词低权重挂上
            if tk:
                keywords.append(f"{tk}^{w}")  # 带上自己的权重

        return MatchTextExpr(self.query_fields, " ".join(keywords), 100, {"minimum_should_match": min(3, round(len(keywords) / 10)), "original_query": " ".join(origin_keywords)})
