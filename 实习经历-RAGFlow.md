# RAGFlow 开源项目 · 实习经历

方向：后端研发（Go / Python）—— Agent 工作流引擎、知识编译与检索
时间：20XX.XX – 20XX.XX（自行填写）
项目：RAGFlow（基于深度文档理解的开源 RAG 引擎，Python / Go 双后端）

## 一、业务介绍

RAGFlow 是基于深度文档理解的开源 RAG（检索增强生成）引擎，业务主链路为：文档上传 → 深度解析（OCR、版面识别、表格结构还原）→ 切片 → 向量化 → 索引（ES / Infinity 等向量 + 全文混合引擎）→ 混合检索与图谱检索 → LLM 流式生成问答。主链之上有三块高级能力：

- Agent 画布工作流：可视化编排 LLM、检索、分支、循环、并行、人机交互等二十余种组件，支持中断恢复与多轮会话记忆；
- 知识编译：对切片做二次编译，产出 structure 知识图谱 / 超图、RAPTOR 递归摘要树、自动生成的 Wiki 页面，用于增强检索与展示；
- Agentic 检索：用 LangGraph 编排多工具、多思考档位的代理式检索（含实时图探索）。

仓库内 Python（Quart）与 Go（Gin）两套后端并行、功能整体对齐，共享 MySQL / Redis / MinIO 与文档引擎，前端（React）运行时探测后端身份做切换。实习期间我主要负责 Go 版 Agent 工作流引擎（基于字节开源的 CloudWeGo eino 框架）：画布编译、组件运行、SSE 流式传输、分布式运行生命周期与检索链路；同时负责 Python 侧的知识编译（Wiki / structure / RAPTOR）与图检索、Agentic RAG 链路。

## 二、技术栈

- Go：Go 1.26、Gin、GORM、CloudWeGo eino（Workflow / ReAct / CheckPoint / StreamReader）、Redis（Lua 原子脚本、分布式锁、检查点存储）、NATS JetStream、SSE 流式推送、CGO 原生解析库与进程内 ONNX Runtime
- Python：Python 3.13、Quart（异步 Web）、Peewee、LangGraph 1.2（StateGraph）、asyncio（信号量 / 线程池 / 多进程 worker）、Redis Streams、tiktoken、ONNX Runtime（OCR / YOLOv10 版面识别 / 表格结构识别）
- 存储与中间件：Elasticsearch / Infinity（向量 + 全文混合检索引擎，共六种可互换后端）、MySQL、Redis、MinIO、Docker Compose
- 前端对接：React + TypeScript（双后端变体机制、Artifacts 展示页与画布对接）

## 三、主要工作与技术难点

### A. Go 版 Agent 工作流引擎（CloudWeGo eino）—— 主负责

1. 画布 DSL → eino 图编译：负责把前端可视化画布 JSON（components / upstream / downstream / parentId）编译为 eino 强类型 Workflow，实现五阶段构建：Loop / Parallel 分组宏展开 → 节点注册（Lambda 节点 + StatePre / StatePost 钩子）→ 边接线（首个上游作数据边、其余上游转显式依赖）→ switch / categorize 多分支接线（分支条件函数读 _next 字段，支持多目标）→ START / END 与合成终端汇聚节点（多终端统一字段映射）。解码器兼容新旧两种画布 JSON 形态，全程零修改 eino 源码。

2. Loop / Parallel 宏节点引擎：eino 原生没有画布式循环 / 并行语义，用「单 Lambda 宏节点 + 内嵌子 Workflow」自研实现。循环：插入合成初始化节点保证循环变量只在首轮初始化一次，循环体直接读写运行级上下文黑板（而非 eino 局部状态），使计数器等变量能跨轮次存续；支持轮次级流式输出（schema.Pipe 桥接）与「迭代粒度」断点恢复，中断异常以组合形式向上重抛，另有 1024 轮安全上限。并行：支持最大并发信号量、并发数 ≤1 时退化为纯串行（零 goroutine 开销）、每个子项独立深拷贝状态、独立 panic 恢复，恢复时只重放已完成结果、仅重调被中断的子项，子项产出经收集节点做数据层合并。

3. ReAct Agent 组件：基于 eino 的 flow/agent/react 构建 ReAct 循环（最大步数 = 轮数 × 2 + 1，按图节点语义折算，默认 5 轮）。自研流式工具调用检查器：多数模型供应商在流末尾才附加 tool_calls，须把模型流排干到 EOF 才能判定工具分支；同时为保证思考 token 实时可见，在判定前用 MessageFuture 启动并发发射器。支持子 Agent 嵌套调用（深度上限 8 层）、工具调用记忆压缩（LLM 摘要成 ≤30 词短句写入长期记忆）、引用接地（答案插入 [ID:N] 标记并标注 grounding 状态）、错误分类分流（取消 / 超时走真实 error，图运行错误转 _ERROR 数据供画布异常分支消费）。

4. 端到端 SSE 流式传输：实现 token 跨越六个异步边界的完整链路——模型驱动层（schema.Pipe + 互斥锁保护的 sender 回调 + [DONE] 哨兵过滤）→ ReAct 并发发射器 → runtime 事件总线（三类 emitter + 四本发射台账做去重）→ service 层 <think> 标签状态机 → Runner 事件通道（有界缓冲）→ Gin SSE（逐帧 Flush、data:[DONE] 收尾、与 Python wire 兼容的扁平事件信封）。区分三种取消生命周期（run ctx / event ctx / HTTP 连接）：客户端断连后后台转「只排空不转发」，运行取消仍保证送达最终 cancelled 事件。Message 组件支持延迟打开 Agent 缓执行流（DeferredStream 占位 + 挂载 sink 逐 token 转发），配合内容一致性比对 + 发射记账双重去重，避免可见输出重复发送。

5. 分布式运行生命周期（Redis 租约 + 跨实例取消）：单条原子 Lua 脚本完成「冲突检查 → 清残留取消标记 → 写租约 → 设 TTL」（租约键 agent:active-session:{sessionID}，TTL 30 秒），续租、释放、发取消标记全部 token 防护（CAS 语义，防误操作他人持有的租约）；看门狗按 TTL/3 节奏续租、未确认续租达 2·TTL/3 判定租约丢失，且挂在 context.WithoutCancel 派生上下文上——客户端断连不会杀死监督 goroutine。跨实例取消靠 cancel 标记 500ms 轮询（取消传播延迟上界约 500ms，由轮询间隔决定）；取消 API 三段解析：本地 run → 远端租约 → 幂等成功（避免误杀未来复用同一 sessionID 的运行）。

6. 检查点与 Human-in-the-loop 断点恢复：Redis 实现 eino 的 CheckPointStore 接口（键 agent:cp:{runID}，TTL 24 小时）；UserFillUp 组件在节点体内抛中断，编译期自动把所有出度 > 0 的非终端节点设为 interrupt-after 候选。设计三层检查点架构：外层 Redis 存储 + 循环 / 并行内存桥存储 + 并行子项级独立 checkpoint ID，支持「循环轮次内、并行子项里的 UserFillUp 中断」这种子节点粒度恢复，恢复时用 ResumeWithData 注入用户填写的答案。CanvasState 手写 JSON 序列化（RWMutex、atomic.Bool 等字段无法走默认序列化）并注册进 eino 内部序列化器表，保证跨进程重启精确还原；取消门控跳过持久化，safeInvoke 保证被取消的运行不会半途弃写。

7. 多轮会话记忆（会话级 DSL 快照）：会话创建时复制画布 DSL，每轮运行优先加载会话行内快照，运行结束「回烤」：把黑板的 Sys / Env / Globals 平铺回带前缀的 globals、history / memory 按 Python wire 格式编码写回、轮次 +1；中断轮的部分答案走同一持久化路径。用户之后编辑画布不影响已有会话记忆；运行失败回滚删除新建的会话行。

8. 双黑板状态管理：eino 每次 Invoke 深拷贝局部状态，而二十余个组件并发改写挂在上下文里的运行级 CanvasState（Outputs / Sys / Env / History / Memory / Retrieval / Globals / CancelFlag，单 RWMutex 保护）。用 StatePre / StatePost 钩子做两个状态平面的双向同步：节点执行前注入上下文快照（history / memory 按「更长者胜」合并），执行后把输出按 组件ID@参数名 平铺；并行 worker 各持状态克隆、绝不共享别名，检索引用按 run 级累计并以 chunk ID 去重（对齐 Python Graph.add_reference 语义）；检查点快照只取读锁，持久化期间组件仍可并发写状态。

9. Go 检索链路（组件 → 工具 → NLP 适配层）：搭建 Retrieval 组件、数据集搜索工具与 NLP 适配层的完整调用链：参数归一化（TopN / 相似度阈值 / 关键词权重 / rerank 模型 / 跨语言 / TOC 增强 / 元数据过滤）、单租户校验、跨数据集 embedding 模型一致性检查、跨语言 query 翻译、父子块扩展（命中子块回溯父块）、chunk 渲染以 [ID:%s] 为每条切片加头部标记，供模型在答案中插入 [ID:N] 引用（与 Python 的引用标记约定一致）；服务层 kb_prompt 的树状模板与字段结构则与 Python 对齐（ID: n + ├── Title: / └── Content:；ID 语义与换行转义细节存微差），检索引用写入状态台账供引用溯源。

10. 引擎层混合检索与打分：默认 KNNTopK=1024、候选 2048、相似度阈值 0.2、向量权重 0.3、pagerank 排序特征、rerank 候选 64；统一打分公式 sim = 文本权重 × 文本相似 + 向量权重 × 向量相似，对 ES / Infinity / OceanBase / 模型重排四条打分路径分别适配（ES 路径含本地计算回退），稳定降序排序并产出文档聚合（doc_aggs）；空结果自动重试（放宽 min_match 至 0.1、向量相似至 0.17）。另把 RAGFlow 自有模型驱动层桥接为 eino 的 ToolCallingChatModel 接口（Generate / Stream / WithTools，WithTools 返回新实例、不改接收者）。

11. 双后端一致性与服务装配：Go 后端（Gin，9384 端口）与 Python 后端（Quart，9380）共享 MySQL / Redis / MinIO / 文档引擎，单二进制四模式（api / admin / ingestor / syncer）。SSE 信封、会话语义乃至历史怪癖都与 Python 线上格式兼容（wire-compatible）（工具名拼写 search_my_dateset、task_id 别名、history / memory JSON 形态），前端读 /api/v1/language 响应体判定后端身份（X-API-Source 响应头是 Go 侧对调用方的统一标记），nginx 三套反代配置（python / golang / hybrid）支持按端点灰度切流。摄取流水线与 Agent 复用同一组件运行时（注册进同一 Registry、共用统一执行入口），同样编译为 eino Workflow 并支持检查点恢复。

### B. Python 侧摄取 / 切片与知识编译（Wiki / structure / RAPTOR）

12. 摄取管线并发架构与分布式计数门禁：任务队列基于 Redis Streams 消费组（XADD / XREADGROUP / XACK，崩溃后回放未 ACK 消息）。三层并发模型：多 worker 进程 × 事件循环本地信号量（LoopLocalSemaphore 用 WeakKeyDictionary 给每个 loop 映射独立信号量）× 单次调用的单线程池（contextvars.copy_context 传递上下文，规避 Python 3.13 下共享线程池的死锁问题）。设计幂等分布式计数门禁（SET NX 去重 + DECRBY 计数，配合 abort 标记 / 重试 ≥3 放弃 / cancel 键三种负反馈）：同一文档被拆成多个分片任务、落在不同进程上时，只有最后成功完成的那个分片才触发 structure 编译与 RAPTOR 的 asyncio.gather 并发执行（两者读同一批切片、写互不相交的 ES 行）。语义为至多一次：计数键缺失、过期或文档已被中止时一律跳过后处理，宁可零次触发也不重复跑。

13. 切片与索引：deepdoc 视觉解析链跑在 ONNX Runtime 上——OCR 文本检测 + 识别、YOLOv10 十类版面识别（页眉 / 页脚 / 参考文献自动丢弃）、表格结构识别把单元格重组为 HTML，OCR 支持 CUDA / CPU 推理与多 GPU 页级分片，版面识别另有 Ascend NPU 与远程 TensorRT DLA 服务路径。按文档类型分发 14 种模板化分块器（naive / paper / book / laws / qa / table / resume / email 等），tiktoken cl100k_base 做 token 预算（默认 512 / 128）+ 分隔符与重叠比例的合并策略；chunk ID 用 xxhash(content + doc_id) 天然幂等，embedding 用文件名向量 ×0.1 + 内容向量 ×0.9 加权混合（filename_embd_weight 可配，写入 q_{dim}_vec），父块先行入库置 available_int=0——对普通检索隐身，仅供父子块扩展按 ID 回捞；同一行 schema 写入六种可互换文档引擎（ES / Infinity / OpenSearch / OceanBase / GaussDB / SereneDB）。

14. RAPTOR 递归摘要树：实现「聚类 → 摘要 → 再聚类」的递归构建：一维相邻余弦「分水岭」百分位聚类（规避完整层次聚类的 O(n²) 距离矩阵），逐层并发 LLM 摘要（带结果缓存与重试），可选物化树形结构；摘要行作为普通 chunk 入库，被混合检索隐式消费，天然支持跨切片的总结型问答。分文档级与知识库级两条产线，带断点续跑与旧摘要清理（按 raptor_kwd 身份标记普查定位旧行，再按 chunk ID 清单精确删除）。

15. structure 知识编译（建图）与增量合并：两阶段 LLM 抽取产出实体 / 关系，支持 list / timeline / hypergraph 等多种结构类型。设计增量合并流水线：同名折叠 → 按结构隔离键分桶、桶内两两余弦相似度归并 → 并查集把实体候选聚成连通分量、整组送 LLM 裁决（相似度只是门票、LLM 是法官：可整组否决原样保留，也可部分合并）→ 别名落定后级联改写关系边端点。链式结构（list / timeline）禁止成环：Tarjan SCC 检测 + LLM 只裁剪不造边，全程 fail-open。边判重设三防线：字节相同边靠稳定行 ID 确定性覆盖、本地阶段不做相似合并、ES 侧端点逐字相等硬过滤 + KNN + LLM 二次裁决。dataset 级合并在 Redis 知识库级锁保护下进行，收尾做 ES KNN 去重与图缓存行重建。产物写为 ES 的 knowledge_graph_kwd = entity / relation / graph / dataset_graph 行（图行 available_int=0，对普通检索隐身），同时被 KGSearch 检索、Agent 的 graph_explore 工具与前端 Artifacts 页消费。

16. Wiki 自动编译与 Artifacts 展示：多阶段增量引擎——增量判定与基线加载 → 入选文档筛选 + 切片级四路 delta（new / changed / deleted / unchanged）→ 模型解析 → 带断点的逐文档 MAP → 增量编译（实体匹配 → REDUCE 裁决 → REFINE 精炼）→ 物化图存储 → 零错误才 commit（版本化检查点支持断点续跑）。REDUCE 裁决给每个候选页定 create / update / delete / noop；规范实体匹配用 KNN + LLM 双阈值（0.90 直判 / 0.75 送 LLM 复核）；两种页面模式：一实体一页模式与主题模式（page-router 用 KNN + LLM 把素材路由到主题页）。finalize 做页面双向互链，产出 wiki_page / wiki_entity / wiki_relation 投影行，经 REST 端点供前端 Artifacts 画布展示页面与知识图谱；Go 后端在 internal/ingestion/component/knowledge_compiler 有对应实现。

### C. 检索与 Agentic RAG

17. 混合检索打分（Python Dealer / Go nlp.Retrieval 双实现）：查询构建阶段做关键词抽取与词项加权（双 IDF × NER × POS）；检索阶段向量 + 全文加权融合（默认向量权重 0.3）、模型重排与本地计算回退、稳定降序排序、相似度阈值截断（向量权重 ≤0 时自动禁用阈值）、文档聚合统计；Go 与 Python 的评分公式与默认参数逐项对齐、服务层 kb_prompt 渲染模板对齐（ID 语义等细节存微差）。

18. KGSearch 图检索：继承混合检索 Dealer 实现专走 knowledge_graph_kwd 字段的图检索管线——从 query 提取实体 → 文档引擎捞实体 / 关系行 → N 跳邻居扩展（按跳数衰减，建图时预计算 pagerank 与 N 跳路径随 chunk 存储）→ 相似度 × pagerank 的贝叶斯组合打分 → 聚合成一段「伪 chunk」参考资料，与普通检索结果一同返回给 LLM。挂载在全局 settings.kg_retriever，接入会话服务等多个问答入口。

19. Agentic RAG（LangGraph）：基于 langgraph 1.2 编排代理式检索——agentic_rag_graph.py 用 StateGraph 构建检索流程，harness/action_session.py 实现「决策节点 + 工具节点」循环。四档思考模式（reasoning 1-4 映射 low / medium / high / ultra）控制工具开放与预算；工具集共八个（结构问答、图探索、目录导航等）。核心工具 graph_explore（ultra 专属）：种子实体 KNN 入选（相似度 ≥0.8、64 候选池、按提及数取前 2）→ 实时两跳 BFS 扩展（from / to_entity_kwd 精确过滤、每跳 32+32、邻居上限 128）→ _ask_structure 让 LLM 基于图证据裁决 → 输出结论或以 source_chunk_ids 回退原文切片作证据；空手而归触发 _disable_tool 熔断禁用该工具，防止会话内反复空耗。navigate_structure 支持零跳直读整图 JSON。Go 侧有同源的 LangGraph 风格移植（internal/harness/graph）。

## 附录：关键模块索引（面试准备用，非简历正文）

- 条1（画布编译）→ internal/agent/canvas/{decode.go, compile.go, scheduler.go, multibranch.go}
- 条2（Loop/Parallel）→ internal/agent/canvas/{loop_subgraph.go, parallel_subgraph.go}、internal/agent/workflowx/{loop.go, parallel.go}
- 条3（ReAct 组件）→ internal/agent/component/agent.go（runEinoReActAgent · scanAllStreamForToolCall · invokeNow）
- 条4（SSE 流式）→ internal/entity/models/llm.go（Stream）、internal/agent/runtime/context.go、internal/service/agent.go（agentMessageDeltaEmitter）、internal/agent/canvas/runner.go、internal/handler/agent.go（RunAgent）、internal/service/bot_completion.go（WriteChatbotRunEvent）、internal/agent/component/message.go（resolveDeferredTemplate）
- 条5（租约/取消）→ internal/agent/canvas/{run_tracker.go, cancel.go}、internal/service/agent.go（RunAgent · CancelSessionRun）
- 条6（检查点/中断）→ internal/agent/canvas/{checkpoint_store.go, interrupt_resume.go}、cmd/ragflow_server.go（buildAgentRunOptions）
- 条7（会话 DSL 快照）→ internal/service/agent.go（persistAgentRunSession · buildPersistedAgentDSL）、internal/agent/canvas/decode.go（EncodeHistory / EncodeMemory）
- 条8（状态管理）→ internal/agent/runtime/state.go（CanvasState）、internal/agent/canvas/scheduler.go（statePre / statePost）
- 条9（Go 检索链）→ internal/agent/component/universe_a_wrappers.go、internal/agent/tool/{retrieval.go, retrieval_nlp.go}
- 条10（引擎层检索）→ internal/service/nlp/retrieval.go、internal/entity/models/llm.go（EinoChatModel）
- 条11（双后端装配）→ cmd/ragflow_server.go、internal/router/{router.go, agent_routes.go}、docker/nginx/ragflow.conf.*
- 条12（摄取并发）→ api/db/services/task_service.py（queue_tasks · 计数门禁）、rag/utils/redis_conn.py、rag/svr/task_executor.py、rag/svr/task_executor_refactor/{task_handler.py, chunk_post_processor.py}、common/asyncio_utils.py（LoopLocalSemaphore）、common/misc_utils.py（thread_pool_exec）
- 条13（切片/索引）→ deepdoc/vision/{ocr.py, layout_recognizer.py, table_structure_recognizer.py}、rag/app/*（14 分块器）、rag/nlp/__init__.py（naive_merge）、common/token_utils.py、rag/utils/{es_conn.py, infinity_conn.py}
- 条14（RAPTOR）→ rag/advanced_rag/knowlege_compile/raptor.py
- 条15（structure 编译）→ rag/advanced_rag/knowlege_compile/{structure.py, runner.py}、api/apps/services/structure_graph_common.py
- 条16（Wiki）→ rag/svr/task_executor_refactor/dataset_wiki_generator.py、rag/advanced_rag/knowlege_compile/{wiki.py, wiki_incremental.py}
- 条17（混合检索）→ rag/nlp/search.py（Dealer）、internal/service/nlp/retrieval.go
- 条18（KGSearch）→ rag/graphrag/search.py（KGSearch，继承 Dealer）、rag/graphrag/utils.py（pagerank / n_hop_with_weight）
- 条19（Agentic RAG）→ rag/advanced_rag/{agentic_rag.py, agentic_rag_graph.py}、rag/advanced_rag/harness/{action_session.py, structure_qa.py}、pyproject.toml（langgraph==1.2.0）
