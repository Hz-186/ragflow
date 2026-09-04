"""GraphRAG 读侧检索器 KGSearch —— 用户提问时，把知识图谱变成一段「参考资料文本」
塞给 LLM 回答。这是整条 GraphRAG 链路的消费端。

一个关键认知：写入侧产出的实体/关系/社区报告全部带着 available_int=0，
普通混合检索（rag/nlp/search.py 的 Dealer）根本搜不到它们 —— 它们只认
KGSearch 这条专门按 knowledge_graph_kwd 字段检索的路。

入口：settings.kg_retriever = KGSearch(docStoreConn)（common/settings.py 初始化），
各问答入口（dialog_service 等）在知识图谱开关打开时调它的 retrieval()。

检索流水线（retrieval 方法的五步）：
    ① 查询改写：把用户问题交给 LLM，抽出「答案类型关键词」和「问题里的实体」
       （类型池来自建图时存的 ty2ents chunk，不是让 LLM 自由发挥）
    ② 三路捞取：
       - 按实体关键词做向量检索，捞 entity chunk（相似度 ≥ 0.3 才收）
       - 按答案类型过滤，捞该类型下 pagerank 最高的 entity
       - 按问题原文做向量检索，捞 relation chunk
    ③ N 跳加成：命中实体自带的 n_hop_with_weight 路径展开，给路径上每段
       边补一个衰减后的相似度（离命中实体越远，加的分越少）
    ④ 打分排序：分数 = 相似度 × pagerank（贝叶斯近似：P(E|Q) ∝ P(E)·P(Q|E)），
       多路同时命中的实体/关系还有额外倍率加成
    ⑤ 组装输出：把实体表、关系表、社区报告拼成一段 CSV 风格文本，
       包进一个「伪 chunk」返回，供上层和普通切片一起喂给 LLM

末尾的 __main__ 块是命令行冒烟测试入口（直接跑本文件可手动验证检索效果）。
"""
import asyncio
import json
import logging
from collections import defaultdict
from copy import deepcopy
import json_repair
import pandas as pd

from common.misc_utils import get_uuid
from rag.graphrag.query_analyze_prompt import PROMPTS
from rag.graphrag.utils import get_entity_type2samples, get_llm_cache, set_llm_cache, get_relation
from common.token_utils import num_tokens_from_string

from rag.nlp.search import Dealer, index_name
from common.float_utils import get_float
from common import settings
from common.doc_store.doc_store_base import OrderByExpr


class KGSearch(Dealer):
    """知识图谱检索器 —— 继承 Dealer 是为了白拿 get_vector（向量查询构造）、
    get_filters（基础过滤条件）等基础设施。"""

    async def _chat(self, llm_bdl, system, history, gen_conf):
        """带缓存的 LLM 调用 —— 查询改写专用（写入侧的 _async_chat 是另一套）。

        缓存键 = 模型名+提示词+历史+参数 的哈希（内容寻址，见 utils.get_llm_cache）：
        同样的问题改写过一次，24 小时内再问直接拿缓存，不再烧 LLM 钱。

        参数长这样：
            llm_bdl = LLMBundle(...)                    # 对话模型
            system  = "---Role---\nYou are..."          # 改写提示词（当系统消息用）
            history = [{"role": "user", "content": "Output:"}]
            gen_conf = {}

        返回值：LLM 的回答文本（期望是一个 JSON 字符串）。
        """
        response = get_llm_cache(llm_bdl.llm_name, system, history, gen_conf)
        if response:
            return response
        response = await llm_bdl.async_chat(system, history, gen_conf)
        # 模型层返回的错误标记：当失败处理，不写缓存
        if response.find("**ERROR**") >= 0:
            raise Exception(response)
        set_llm_cache(llm_bdl.llm_name, system, response, history, gen_conf)
        return response

    async def query_rewrite(self, llm, question, idxnms, kb_ids):
        """查询改写：用户问题 → (答案类型关键词, 问题里的实体) 两组检索线索。

        推演：
            输入  question = "姚明效力过哪些球队？"
            第 1 步：查类型池 —— get_entity_type2samples 把建图时存的
                     ty2ents chunk 捞回来，形如 {"ORGANIZATION": ["NBA", ...], "PERSON": ["姚明", ...]}
            第 2 步：套 minirag_query2kwd 模板（问题 + 类型池）问 LLM
            第 3 步：解析回答的 JSON（json_repair 能容忍缺引号之类的小毛病）
            输出  (["ORGANIZATION"], ["姚明", "球队"])
                  # 类型最多 3 个；实体只取前 5 个

        解析彻底失败时向上抛异常，调用方（retrieval）兜底：把问题原文当实体用。
        """
        ty2ents = await get_entity_type2samples(idxnms, kb_ids)
        # 把类型池转成缩进 JSON 文本塞进模板，让 LLM「照着池子选」
        hint_prompt = PROMPTS["minirag_query2kwd"].format(query=question, TYPE_POOL=json.dumps(ty2ents, ensure_ascii=False, indent=2))
        result = await self._chat(llm, hint_prompt, [{"role": "user", "content": "Output:"}], {})
        try:
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            # 实体线索只取前 5 个，防止问题里塞了一长串名词把检索带偏
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]
            return type_keywords, entities_from_query
        except json_repair.JSONDecodeError:
            try:
                # 抢救式解析：有的模型会把整段提示词复读回来，只在尾巴吐出 JSON。
                # 这里剥掉复读的提示词和角色标记，再抠出第一对 {...} 重新解析
                result = result.replace(hint_prompt[:-1], "").replace("user", "").replace("model", "").strip()
                result = "{" + result.split("{")[1].split("}")[0] + "}"
                keywords_data = json_repair.loads(result)
                type_keywords = keywords_data.get("answer_type_keywords", [])
                entities_from_query = keywords_data.get("entities_from_query", [])[:5]
                return type_keywords, entities_from_query
            # 抢救也失败：记日志，把解析错误抛给调用方兜底
            except Exception as e:
                logging.exception(f"JSON parsing error: {result} -> {e}")
                raise e

    def _ent_info_from_(self, es_res, sim_thr=0.3):
        """把实体检索结果整理成字典 —— 每命中一个实体，收一份「实体档案」。

        参数长这样：
            es_res = 文档引擎的检索返回（含命中的 entity chunk）
            sim_thr = 0.3   # 相似度门槛，低于它的命中直接丢弃

        返回值长这样（键 = 实体名）：
            {
                "姚明": {
                    "sim": 0.42,            # 向量相似度（引擎返回的 _score）
                    "pagerank": 0.0032,     # 重要性分数（写入侧算好的 rank_flt）
                    "n_hop_ents": [          # N 跳路径（建图时预存）
                        {"path": ["姚明", "NBA"], "weights": [3.0]}, ...],
                    "description": '{"entity_type": "PERSON", ...}',  # 实体档案 JSON 原文
                },
            }
        """
        res = {}
        flds = ["content_with_weight", "_score", "entity_kwd", "rank_flt", "n_hop_with_weight"]
        es_res = self.dataStore.get_fields(es_res, flds)
        for _, ent in es_res.items():
            # 字段值为 None 的键先删掉，避免后面误用
            for f in flds:
                if f in ent and ent[f] is None:
                    del ent[f]
            # 相似度不达标的命中：不收
            if get_float(ent.get("_score", 0)) < sim_thr:
                continue
            # entity_kwd 偶尔以列表形式回来，取第一个
            if isinstance(ent["entity_kwd"], list):
                ent["entity_kwd"] = ent["entity_kwd"][0]
            # n_hop_with_weight 两种坑：老数据没这个字段；Infinity 引擎的
            # 列默认值是空字符串——两种都没法直接 json.loads，统一兜底成 "[]"
            n_hop_raw = ent.get("n_hop_with_weight") or "[]"
            try:
                n_hop_ents = json.loads(n_hop_raw)
            except (json.JSONDecodeError, TypeError):
                logging.warning(f"Failed to parse n_hop_with_weight for entity {ent.get('entity_kwd')}: {n_hop_raw}")
                n_hop_ents = []
            res[ent["entity_kwd"]] = {
                "sim": get_float(ent.get("_score", 0)),
                "pagerank": get_float(ent.get("rank_flt", 0)),
                "n_hop_ents": n_hop_ents,
                "description": ent.get("content_with_weight", "{}"),
            }
        return res

    def _relation_info_from_(self, es_res, sim_thr=0.3):
        """把关系检索结果整理成字典 —— 与上面的实体版对称。

        返回值长这样（键 = 排好序的端点对，保证 (A,B)/(B,A) 同一键）：
            {
                ("NBA", "姚明"): {
                    "sim": 0.35,
                    "pagerank": 3.0,     # 注意：关系这边用的是 weight_int（边权重），
                                         # 字段名叫 pagerank 只是复用同一个打分槽位
                    "description": '{"weight": 3.0, "description": "...", ...}',
                },
            }
        """
        res = {}
        es_res = self.dataStore.get_fields(es_res, ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd", "weight_int"])
        for _, ent in es_res.items():
            if get_float(ent.get("_score", 0)) < sim_thr:
                continue
            # 端点排序：与写入侧（handle_single_relationship_extraction 的 sorted）
            # 保持同一种写法，避免同一条边两个键
            f, t = sorted([ent["from_entity_kwd"], ent["to_entity_kwd"]])
            if isinstance(f, list):
                f = f[0]
            if isinstance(t, list):
                t = t[0]
            res[(f, t)] = {"sim": get_float(ent.get("_score", 0)), "pagerank": get_float(ent.get("weight_int", 0)), "description": ent["content_with_weight"]}
        return res

    async def get_relevant_ents_by_keywords(self, keywords, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        """按实体关键词做向量检索，捞最相关的实体 —— 三路捞取之「关键词路」。

        参数长这样：
            keywords = ["姚明", "球队"]      # 查询改写抽出的实体线索
            filters  = {"kb_id": [...]}     # 基础过滤，本函数追加 entity 身份过滤
            emb_mdl  = LLMBundle(向量模型)

        工作方式：关键词用 ", " 拼成一句话 → 向量化 → 对 entity 类 chunk
        做 KNN 检索（top_k=1024，相似度下限 sim_thr）→ 取前 N=56 条 →
        交给 _ent_info_from_ 整理成档案字典。

        返回值：{"姚明": {...档案...}, ...}；关键词为空返回 {}。
        """
        if not keywords:
            return {}
        filters = deepcopy(filters)  # 不改调用方的原字典
        filters["knowledge_graph_kwd"] = "entity"  # 只捞实体类 chunk
        matchDense = await self.get_vector(", ".join(keywords), emb_mdl, top_k=1024, num_candidates=2048, similarity=sim_thr)
        es_res = self.dataStore.search(["content_with_weight", "entity_kwd", "rank_flt", "n_hop_with_weight"], [], filters, [matchDense], OrderByExpr(), 0, N, idxnms, kb_ids)
        return self._ent_info_from_(es_res, sim_thr)

    async def get_relevant_relations_by_txt(self, txt, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        """按文本做向量检索，捞最相关的关系 —— 三路捞取之「问题原文路」。

        与实体版的区别：检索对象换成 relation 类 chunk（它们的向量是
        "A->B: 描述" 算出来的，见 utils.graph_edge_to_chunk），其余同理。

        返回值：{("NBA", "姚明"): {...档案...}, ...}；文本为空返回 {}。
        """
        if not txt:
            return {}
        filters = deepcopy(filters)
        filters["knowledge_graph_kwd"] = "relation"
        matchDense = await self.get_vector(txt, emb_mdl, top_k=1024, num_candidates=2048, similarity=sim_thr)
        es_res = self.dataStore.search(["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd", "weight_int"], [], filters, [matchDense], OrderByExpr(), 0, N, idxnms, kb_ids)
        return self._relation_info_from_(es_res, sim_thr)

    def get_relevant_ents_by_types(self, types, filters, idxnms, kb_ids, N=56):
        """按实体类型过滤，捞该类型下最重要的实体 —— 三路捞取之「类型路」。

        与前两路的本质区别：这一路不做向量检索（没有语义相似度），纯按
        entity_type_kwd 过滤 + pagerank 降序——即「这个类型里最核心的实体是谁」。

        参数长这样：
            types = ["ORGANIZATION", "PERSON"]   # 查询改写选出的答案类型
            N = 10000                            # retrieval 里传的就是这个数，几乎全收

        注意：_ent_info_from_ 这里传阈值 0 —— 类型路没有相似度概念，全收。

        返回值：{"NBA": {...档案...}, ...}；类型为空返回 {}。
        """
        if not types:
            return {}
        filters = deepcopy(filters)
        filters["knowledge_graph_kwd"] = "entity"
        filters["entity_type_kwd"] = types
        # 按 pagerank（rank_flt）降序：类型里的「大人物」排前面
        ordr = OrderByExpr()
        ordr.desc("rank_flt")
        es_res = self.dataStore.search(["entity_kwd", "rank_flt"], [], filters, [], ordr, 0, N, idxnms, kb_ids)
        return self._ent_info_from_(es_res, 0)

    async def retrieval(
        self,
        question: str,
        tenant_ids: str | list[str],
        kb_ids: list[str],
        emb_mdl,
        llm,
        max_token: int = 8196,
        ent_topn: int = 6,
        rel_topn: int = 6,
        comm_topn: int = 1,
        ent_sim_threshold: float = 0.3,
        rel_sim_threshold: float = 0.3,
        **kwargs,
    ):
        """图谱检索总入口 —— 一个问题进来，一段「图谱参考资料」出去。

        参数长这样：
            question = "姚明效力过哪些球队？"
            tenant_ids = "tenant_abc"      # 逗号分隔字符串或列表均可
            kb_ids = ["kb_001"]
            emb_mdl = LLMBundle(向量模型)   # 给查询做向量化
            llm = LLMBundle(对话模型)       # 查询改写用
            max_token = 8196               # 产出文本的 token 预算（实体/关系/报告共享）
            ent_topn / rel_topn / comm_topn = 6 / 6 / 1   # 实体/关系/社区报告各取前几
            ent_sim_threshold / rel_sim_threshold = 0.3   # 两路向量检索的相似度门槛

        返回值是一个「伪 chunk」—— 长得像普通切片，实则是把图谱内容拼成的文本：
            {
                "chunk_id": "uuid...",
                "content_ltks": "",                # 没有分词（不参与文本打分）
                "content_with_weight": "\\n---- Entities ----\\nEntity,Score,Description\\n...
                    \\n---- Relations ----\\nFrom Entity,To Entity,Score,Description\\n...
                    \\n---- Community Report ----\\n# 1. Community 0: ...\\n...",
                "doc_id": "",
                "docnm_kwd": "Related content in Knowledge Graph",
                "kb_id": ["kb_001"],
                "similarity": 1.0,                 # 固定值：图谱内容不走普通打分
                "vector": [], "positions": [],
                ...
            }
        上层把这段文本和其他普通切片一起拼进 LLM 的上下文，LLM 回答时就能
        引用跨段落串联的图谱信息。
        """
        qst = question
        filters = self.get_filters({"kb_ids": kb_ids})  # 基础过滤（限定知识库范围）
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")
        # 每个租户一个索引名，多租户一起查
        idxnms = [index_name(tid) for tid in tenant_ids]
        ty_kwds = []
        try:
            # ── 第 ① 步：查询改写 → 类型关键词 + 实体线索 ──
            ty_kwds, ents = await self.query_rewrite(llm, qst, [index_name(tid) for tid in tenant_ids], kb_ids)
            logging.info(f"Q: {qst}, Types: {ty_kwds}, Entities: {ents}")
        except Exception as e:
            # 改写失败（模型挂了/JSON 彻底解析不出）兜底：拿问题原文当实体线索，
            # 图谱检索降级但不停摆
            logging.exception(e)
            ents = [qst]
            pass

        # ── 第 ② 步：三路捞取 ──
        # 关键词路：实体线索向量检索
        ents_from_query = await self.get_relevant_ents_by_keywords(ents, filters, idxnms, kb_ids, emb_mdl, ent_sim_threshold)
        # 类型路：答案类型过滤 + pagerank 排序（大库几乎全收，靠后面排序裁剪）
        ents_from_types = self.get_relevant_ents_by_types(ty_kwds, filters, idxnms, kb_ids, 10000)
        # 原文路：问题全文向量检索关系
        rels_from_txt = await self.get_relevant_relations_by_txt(qst, filters, idxnms, kb_ids, emb_mdl, rel_sim_threshold)
        # ── 第 ③ 步：N 跳加成 —— 命中实体的邻居路径也给相邻的边加分 ──
        # nhop_pathes：{(端点1, 端点2): {"sim": 累计相似度, "pagerank": 边权最大值}}
        nhop_pathes = defaultdict(dict)
        for _, ent in ents_from_query.items():
            nhops = ent.get("n_hop_ents", [])
            if not isinstance(nhops, list):
                logging.warning(f"Abnormal n_hop_ents: {nhops}")
                continue
            for nbr in nhops:
                path = nbr["path"]    # 形如 ["姚明", "NBA", "火箭队"]
                wts = nbr["weights"]  # 形如 [3.0, 1.0]（每段边的权重）
                # 路径逐段拆开，每段边都分得一份衰减相似度：
                # 第 1 段除以 2、第 2 段除以 3……离命中实体越远加得越少
                for i in range(len(path) - 1):
                    f, t = path[i], path[i + 1]
                    if (f, t) in nhop_pathes:
                        nhop_pathes[(f, t)]["sim"] += ent["sim"] / (2 + i)
                    else:
                        nhop_pathes[(f, t)]["sim"] = ent["sim"] / (2 + i)
                    # 边权重取所有来源里的最大值（多条路径重复经过同一段边时）
                    nhop_pathes[(f, t)]["pagerank"] = max(nhop_pathes[(f, t)].get("pagerank", 0), wts[i])

        logging.info("Retrieved entities: {}".format(list(ents_from_query.keys())))
        logging.info("Retrieved relations: {}".format(list(rels_from_txt.keys())))
        logging.info("Retrieved entities from types({}): {}".format(ty_kwds, list(ents_from_types.keys())))
        logging.info("Retrieved N-hops: {}".format(list(nhop_pathes.keys())))

        # ── 第 ④ 步：打分 —— 贝叶斯近似 P(E|Q) ∝ P(E) × P(Q|E)，
        # 即 pagerank（实体的先天重要性）× sim（查询与实体的匹配度）──
        # 双重命中加成：实体既被关键词路命中、又是答案类型的核心成员 → 相似度翻倍
        for ent in ents_from_types.keys():
            if ent not in ents_from_query:
                continue
            ents_from_query[ent]["sim"] *= 2

        # 关系加成：每条被原文路命中的关系，按「证据数量」放大相似度
        for f, t in rels_from_txt.keys():
            pair = tuple(sorted([f, t]))
            s = 0
            if pair in nhop_pathes:
                # 证据 1：N 跳路径也指向这条边 → 加上路径相似度，并从
                # nhop_pathes 里移除（已经兑现，下面不再重复计）
                s += nhop_pathes[pair]["sim"]
                del nhop_pathes[pair]
            # 证据 2/3：两端实体各自在答案类型名单里 → 各加 1
            if f in ents_from_types:
                s += 1
            if t in ents_from_types:
                s += 1
            # 乘数 = 证据分 + 1（没有任何证据也保底 ×1 不减分）
            rels_from_txt[(f, t)]["sim"] *= s + 1

        # 补漏：N 跳路径指向的边里，没被原文路直接命中的，也补成一条关系候选
        for f, t in nhop_pathes.keys():
            s = 0
            if f in ents_from_types:
                s += 1
            if t in ents_from_types:
                s += 1
            rels_from_txt[(f, t)] = {"sim": nhop_pathes[(f, t)]["sim"] * (s + 1), "pagerank": nhop_pathes[(f, t)]["pagerank"]}

        # 总分 = sim × pagerank，降序取前 N
        ents_from_query = sorted(ents_from_query.items(), key=lambda x: x[1]["sim"] * x[1]["pagerank"], reverse=True)[:ent_topn]
        rels_from_txt = sorted(rels_from_txt.items(), key=lambda x: x[1]["sim"] * x[1]["pagerank"], reverse=True)[:rel_topn]

        # ── 第 ⑤ 步：组装输出文本（边组装边扣 token 预算，超了就截断）──
        ents = []
        relas = []
        for n, ent in ents_from_query:
            # 实体行：名字 + 分数 + 描述（从档案 JSON 里抠出 description 字段）
            ents.append({"Entity": n, "Score": "%.2f" % (ent["sim"] * ent["pagerank"]), "Description": json.loads(ent["description"]).get("description", "") if ent["description"] else ""})
            max_token -= num_tokens_from_string(str(ents[-1]))
            if max_token <= 0:
                # 预算花光：刚加的这条也放不下，回退一条后截断
                ents = ents[:-1]
                break

        for (f, t), rel in rels_from_txt:
            if not rel.get("description"):
                # N 跳补进来的关系只有端点没有描述：回文档引擎按端点补查一次
                for tid in tenant_ids:
                    rela = await get_relation(tid, kb_ids, f, t)
                    if rela:
                        break
                else:
                    # 所有租户都查不到描述：这条边没法呈现，放弃
                    continue
                rel["description"] = rela["description"]
            desc = rel["description"]
            try:
                # 描述一般是 JSON 包裹的（{"weight":..., "description": "..."}），
                # 抠出内层文本；抠不出就当纯文本用
                desc = json.loads(desc).get("description", "")
            except Exception:
                pass
            relas.append({"From Entity": f, "To Entity": t, "Score": "%.2f" % (rel["sim"] * rel["pagerank"]), "Description": desc})
            max_token -= num_tokens_from_string(str(relas[-1]))
            if max_token <= 0:
                relas = relas[:-1]
                break

        # 实体表/关系表各自转成 CSV 文本（pandas 代劳，表头自动带上）
        if ents:
            ents = "\n---- Entities ----\n{}".format(pd.DataFrame(ents).to_csv())
        else:
            ents = ""
        if relas:
            relas = "\n---- Relations ----\n{}".format(pd.DataFrame(relas).to_csv())
        else:
            relas = ""

        # 三段文本拼进一个「伪 chunk」返回：实体表 + 关系表 + 社区报告
        return {
            "chunk_id": get_uuid(),
            "content_ltks": "",
            "content_with_weight": ents + relas + self._community_retrieval_([n for n, _ in ents_from_query], filters, kb_ids, idxnms, comm_topn, max_token),
            "doc_id": "",
            "docnm_kwd": "Related content in Knowledge Graph",
            "kb_id": kb_ids,
            "important_kwd": [],
            "image_id": "",
            "similarity": 1.0,
            "vector_similarity": 1.0,
            "term_similarity": 0,
            "vector": [],
            "positions": [],
        }

    def _community_retrieval_(self, entities, condition, kb_ids, idxnms, topn, max_token):
        """社区报告检索 —— 给实体表再配一段「高层综述」。

        思路：拿本轮命中的实体名单，去 community_report 类 chunk 里找
        「成员包含这些实体的社区报告」，按社区权重降序取前 topn 篇。

        参数长这样：
            entities = ["姚明", "NBA"]        # 本轮命中的实体名列表
            condition = {"kb_id": [...]}      # 基础过滤，本函数追加社区报告身份过滤
            topn = 1                          # 默认只要权重最高的一篇
            max_token = 剩余 token 预算

        返回值是格式化文本（没有命中时返回空串 ""）：
            "\n---- Community Report ----\n
             # 1. Community 0: 姚明的职业生涯\n
             ## Content\n（LLM 写的社区综述）\n
             ## Evidences\n（支撑证据）\n"
        """
        fields = ["docnm_kwd", "content_with_weight"]
        # 按社区权重降序：成员覆盖面最广、权重最高的社区优先
        odr = OrderByExpr()
        odr.desc("weight_flt")
        fltr = deepcopy(condition)
        fltr["knowledge_graph_kwd"] = "community_report"
        # entities_kwd 存的是社区成员名单：只要与本次命中实体有交集就算匹配
        fltr["entities_kwd"] = entities
        comm_res = self.dataStore.search(fields, [], fltr, [], odr, 0, topn, idxnms, kb_ids)
        comm_res_fields = self.dataStore.get_fields(comm_res, fields)
        txts = []
        for ii, (_, row) in enumerate(comm_res_fields.items()):
            # 报告正文建图时存的 JSON：{"report": "综述", "evidences": "证据"}
            obj = json.loads(row["content_with_weight"])
            txts.append("# {}. {}\n## Content\n{}\n## Evidences\n{}\n".format(ii + 1, row["docnm_kwd"], obj["report"], obj["evidences"]))
            max_token -= num_tokens_from_string(str(txts[-1]))

        if not txts:
            return ""
        return "\n---- Community Report ----\n" + "\n".join(txts)


# ── 命令行冒烟测试入口：直接 `python -m rag.graphrag.search -t 租户 -d 知识库 -q 问题`
# 可以手动验证某个知识库的图谱检索效果，输出就是返回给上层的那个伪 chunk ──
if __name__ == "__main__":
    import argparse
    from common.constants import LLMType
    from api.db.services.knowledgebase_service import KnowledgebaseService
    from api.db.services.llm_service import LLMBundle
    from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type, resolve_model_config, get_model_config_by_id
    from rag.nlp import search

    settings.init_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tenant_id", default=False, help="Tenant ID", action="store", required=True)
    parser.add_argument("-d", "--kb_id", default=False, help="Knowledge base ID", action="store", required=True)
    parser.add_argument("-q", "--question", default=False, help="Question", action="store", required=True)
    args = parser.parse_args()

    kb_id = args.kb_id
    # 对话模型：取租户默认的聊天模型
    llm_config = get_tenant_default_model_by_type(args.tenant_id, LLMType.CHAT)
    llm_bdl = LLMBundle(args.tenant_id, llm_config)
    # 向量模型：优先知识库自己指定的（租户级绑定），没有就用知识库配置的，
    # 再按 id 查不到配置就退回名字解析（兼容新旧两种配置方式）
    _, kb = KnowledgebaseService.get_by_id(kb_id)
    if kb.tenant_embd_id:
        try:
            embd_model_config = get_model_config_by_id(args.tenant_id, LLMType.EMBEDDING, kb.tenant_embd_id)
        except LookupError:
            embd_model_config = resolve_model_config(args.tenant_id, LLMType.EMBEDDING, kb.embd_id)
    else:
        embd_model_config = resolve_model_config(args.tenant_id, LLMType.EMBEDDING, kb.embd_id)
    embed_bdl = LLMBundle(args.tenant_id, embd_model_config)

    kg = KGSearch(settings.docStoreConn)
    print(asyncio.run(kg.retrieval({"question": args.question, "kb_ids": [kb_id]}, search.index_name(kb.tenant_id), [kb_id], embed_bdl, llm_bdl)))
