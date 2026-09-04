#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
"""
RAGFlow 的通用小工具合集。本文件里有这些东西：
    - get_uuid：生成唯一 ID；
    - download_img：下载第三方登录头像（协程版，带 SSRF 防护）；
    - hash_str2int / convert_bytes：字符串转整数哈希、字节数转人类可读文本；
    - once / pip_install_torch：「只执行一次」装饰器、自动安装 PyTorch；
    - thread_pool_exec / thread_pool_exec_long_time：把「阻塞函数」交给工作线程
      去跑的两个桥梁（本文件的重点）；
    - hashable_key 三件套：把任意值变成能当 dict 键的东西（去重用）。

========================================================================
【零基础前置课】进程、线程、协程、GIL —— 读懂本文件前必须知道的背景
========================================================================
（已经懂这些概念的读者，直接跳到下一节【本文件的三条调度路线】。）

1) 进程（process）
   一个「正在运行的程序实例」。比如执行 `python api/ragflow_server.py` 就得到
   一个进程。进程之间的内存完全隔离：A 进程的变量，B 进程看不见。
   RAGFlow 后端本身就是多进程结构：
     - ragflow_server：API 服务器进程，负责接所有前端 HTTP 请求；
     - task_executor（rag/svr/task_executor.py）：常驻的解析干活进程。启动
       脚本按 WS 个拉起（默认 1 个），每个进程不停地从任务队列取任务，
       一个进程内最多同时跑 MAX_CONCURRENT_TASKS 个任务（默认 5）。
       解析、切块、向量化这类重活都在这些进程里做。
   每个 Python 进程都有自己独立的一把 GIL（见下），进程之间互不抢锁。

2) 线程（thread）
   同一个进程内部的多条「执行流」，共享这个进程的内存。线程由操作系统负责
   调度（谁上 CPU 由 OS 说了算）。你可以开很多线程，但在 CPython（大家平时
   用的主流 Python 实现）里受下面的 GIL 限制。

3) GIL（全局解释器锁，Global Interpreter Lock）
   CPython 解释器里的一把全局锁：任何瞬间，最多只有一个线程在执行 Python
   代码。就算你开了 8 个线程，同一瞬间也只有 1 个在跑 Python 代码。
   那线程还有什么用？关键在一条重要例外：线程在「等待 I/O」（等网络响应、
   等磁盘读写）时会主动释放 GIL，让别的线程先跑。于是：
     - CPU 密集活儿（算向量之类）：Python 线程帮不上忙，所以 RAGFlow 把
       解析工作放进独立的 task_executor 进程，而不是靠多线程；
     - I/O 密集活儿（删大文件、等数据库、等网络）：线程非常有用——线程 A
       在等磁盘时，线程 B 正好执行 Python 代码，两边都不闲着。
   本文件末尾的 thread_pool_exec / thread_pool_exec_long_time 就是为了把
   「I/O 密集的同步代码」丢给工作线程而存在的。

4) 协程（coroutine，就是 async/await 这套语法）
   比线程更轻量的「执行流」，由程序自己调度，不经过操作系统。所有协程都跑在
   同一条线程上：这条线程里有个叫「事件循环」（event loop）的总调度，协程 A
   一执行到 await（意思是「我在等一个外部结果，比如等网络」）就主动让出执行权，
   事件循环立刻切去执行协程 B。所以一条线程能同时「伺候」成千上万个协程——
   当然，同一瞬间真正执行 Python 代码的还是只有一个。
   RAGFlow 的 API 服务器（Quart 框架）就是这么工作的：每个 HTTP 请求进来
   就是一个协程，整个服务靠一条事件循环线程轮流处理所有请求。

5) 为什么不全部用协程？
   协程只有搭配「支持 await 的库」（异步库，比如下面 download_img 用的
   httpx）才能发挥威力。但仓库里有大量「同步阻塞」代码：MinIO 对象存储 SDK、
   peewee 数据库库、os.remove() 删文件、嵌入模型 SDK 等等——它们内部根本不
   认识 await。要是在协程里直接调它们，唯一的事件循环线程就被卡死了：相当于
   一个人堵住了唯一的过道，所有 HTTP 请求一起冻结。
   标准解法：把这类阻塞活儿丢给工作线程去跑。那条线程老老实实被卡住没关系，
   事件循环这条主线程腾出手来继续接别的请求。本文件末尾的两个函数就是干这个的。

========================================================================
【本文件的三条调度路线】
========================================================================

路线 A：纯协程 —— download_img
    HTTP 请求协程 --await--> download_img --await--> httpx 异步网络 I/O
    全程不开线程。等网络时协程让出执行权，事件循环去处理别的请求。

路线 B：协程 + 临时工作线程 —— thread_pool_exec
    HTTP 请求协程 --await--> thread_pool_exec --把阻塞函数递进--> 临时工作线程
    每调用一次就新建一个只有 1 条线程的临时线程池，用完即拆。适合「很快就
    干完」的阻塞活儿（典型用法：API 接口里同步调用嵌入模型算向量、
    structure_graph_common 里的同步查询）。

路线 C：协程 + 全局长命工作池 —— thread_pool_exec_long_time
    HTTP 请求协程 --await--> thread_pool_exec_long_time --递进--> 全局工作池
    （池子就是本文件开头创建的 _LONG_TIME_THREAD_POOL_EXECUTOR，默认只有
    1 条线程，可用环境变量 LONG_TIME_THREAD_POOL_WORKERS 调；活儿多了排队）
    即使 HTTP 请求中途被取消、客户端断开连接，工作线程里的活儿也不会被打断——
    删除类任务（FileService.delete_docs、dataset_api_service 里的
    _delete_datasets_sync）必须跑完，否则会留下垃圾数据。

同一时刻到底在跑几个线程、几个协程？（ragflow_server 进程，默认配置）
    - 事件循环主线程：1 条。所有 HTTP 请求协程都在这一条线程上轮流执行；
    - 临时工作线程：同时有 N 个 thread_pool_exec 在等结果，就有 N 条
      （每次调用自己开一个单线程池）；
    - 长命工作线程：1 条（默认值）。同时来两个删除大任务时，第二个排队等；
    - 协程：数量 = 还没处理完的 HTTP 请求数，没有固定上限，全部挤在事件
      循环那一条线程上轮流跑；
    - GIL 的保证：上面这些线程里，任何瞬间只有 1 条在执行 Python 代码，
      其余的要么在等 I/O（此时不占 GIL），要么在排队等 GIL；
    - 补充：服务器进程里另有少数与本文件无关的常驻后台线程（进度更新、
      聊天通道等），不影响上面的大局。
有进程吗？有——ragflow_server 和各个 task_executor 就是不同的进程，各有各
的 GIL；但本文件自己不创建进程，唯一的例外是 pip_install_torch 会用
subprocess 起一个 pip 子进程装包。

========================================================================
【语法小抄】本文件出现的 Python 语法，一句话解释
========================================================================
    async def f():   定义「协程函数」。调用它不会立刻执行，只是返回一个
                     「协程对象」；要被事件循环 await 才真正开始跑。
    await x          「等 x 出结果」。等待期间当前协程让出执行权，事件循环
                     去跑别的协程。只能用在 async def 里面。
    *args            把剩余的位置参数收集成元组：调用 f(1, 2) 时 args == (1, 2)。
    **kwargs         把剩余的关键字参数收集成字典：调用 f(k=3) 时
                     kwargs == {"k": 3}。
    @once            装饰器语法，等价于 func = once(func)：把原函数替换成
                     once 返回的包装函数。
    nonlocal x       在嵌套函数里声明「我要改的 x 是外层函数的那个，别给我
                     新建一个局部的」。
    with x:          上下文管理器：离开这个代码块时自动执行清理（关文件、
                     关线程池等），出了异常也会清理。
    async with / async for    协程里用的 with / for 版本，进入、退出、每轮
                     迭代都可能 await（涉及异步 I/O）。
    functools.partial(f, a)   「预装参数」：返回一个新函数，相当于「已经帮你
                     填好参数 a 的 f」。
    contextvars      「当前请求上下文里才可见的变量」（比如链路追踪 ID）。
                     跨线程不会自动带过去，要手动复制快照（copy_context）。
    ThreadPoolExecutor        线程池：预先养着若干工作线程，你把活儿递进去，
                     由它们代跑。
    frozenset        不可变的集合。能当 dict 键 / set 元素（普通 set 不行，
                     因为它可变、不可哈希）。
    __slots__        声明「本类实例只允许有这几个属性」，省内存、禁止乱加。
    __eq__ / __hash__   Python 判断「两个对象是否相等」和「计算哈希码」的
                     约定方法。规则：相等的对象，哈希码必须也相等。
"""

import asyncio
import base64
import contextvars
import functools
import hashlib
import logging
import os
import subprocess
import sys
import threading
import uuid
from urllib.parse import urljoin

from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
# 全局「长任务工作池」：供下文 thread_pool_exec_long_time 使用。
# 这一行在模块被 import 时就执行（模块顶层代码 = 导入即运行），池子从此常驻，
# 活多久和进程一样久。
# - max_workers = 池里养几条工作线程，由环境变量 LONG_TIME_THREAD_POOL_WORKERS
#   控制，默认 1 条：同时来两个删除大任务时，第二个老实排队，防止无限开线程；
# - thread_name_prefix：给工作线程的名字加 "long-time" 前缀，日志里一眼能认出。
_LONG_TIME_THREAD_POOL_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.getenv("LONG_TIME_THREAD_POOL_WORKERS", "1")), thread_name_prefix="long-time")


def get_uuid():
    """生成一个全局唯一 ID 字符串，全仓库建用户/知识库/文档等主键时都在用它。

    返回值长这样：
        "6ba7b8109dad11d180b400c04fd430c8"
    （32 个十六进制字符。uuid1 = 用时间戳 + 机器 MAC 地址生成，天然按时间有序。）
    """
    return uuid.uuid1().hex


# ============================ OAuth 头像下载的配置 ============================
# 下面的 download_img 用来拉第三方登录（OAuth）的用户头像。这里给它上三道保险：
# 响应体大小有上限、重定向跳数有上限，且每一跳都要过 SSRF 检查 + DNS 锁定
# （SSRF = 服务端请求伪造，防止攻击者借我们的服务器去访问内网；实现见
# common/ssrf_guard.py）。三个值都可以用环境变量覆盖。
_OAUTH_AVATAR_MAX_BYTES = int(os.environ.get("RAGFLOW_OAUTH_AVATAR_MAX_BYTES", str(5 * 1024 * 1024)))  # 响应体最多 5MB
_OAUTH_AVATAR_MAX_REDIRECTS = int(os.environ.get("RAGFLOW_OAUTH_AVATAR_MAX_REDIRECTS", "5"))  # 最多允许重定向 5 跳
# frozenset = 不可变的集合（语法见文件头小抄）。「表示重定向」的 HTTP 状态码
# 集合，后面用 `状态码 in _REDIRECT_STATUS` 判断，集合查询是 O(1) 很快。
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})


async def download_img(url):
    """把给定 URL 的图片下载下来，返回「data URI」字符串（图片以 base64 直接
    嵌在字符串里）；失败或被 SSRF 安全检查拦截时返回空字符串。

    参数：
        url：图片地址，例如 "https://avatar.example.com/u/123.png"。
            允许带重定向。None / 空值直接返回 ""；非空的非字符串会先被
            转成字符串再往下走。

    返回值长这样：
        成功 → "data:image/png;base64,iVBORw0KGgo..."
               （格式：data:<内容类型>;base64,<图片内容的 base64 编码>，
               前端拿到可直接当图片显示）
        失败 → ""

    安全背景：本函数用于 OAuth 第三方登录头像下载（调用方见 user_api）。
    头像 URL 里可能带着登录令牌，所以任何日志都不允许打印 URL 本身；每一跳
    重定向都要先过 SSRF 检查，防止攻击者借这个函数探测内网。
    """
    # --- 第一步：清洗入参。空值、非字符串、去掉空格后为空的，直接返回 "" ---
    if not url:
        return ""
    if not isinstance(url, str):
        url = str(url)
    url = url.strip()
    if not url:
        return ""

    # current_url = 本次要请求的地址（每跟随一次重定向就换成新地址）；
    # redirect_hops = 已经跟了几跳重定向（hop = 跳，重定向一次算一跳）。
    current_url = url
    redirect_hops = 0

    # 从环境变量读 HTTP 配置（超时/代理/UA），默认值与 common/http_client.py
    # 保持一致。这里故意不 import http_client：导入它会把整个配置模块一起
    # 拖进来，保持本函数零依赖，轻量测试环境也能用。
    request_timeout = float(os.environ.get("HTTP_CLIENT_TIMEOUT", "15"))
    proxy = os.environ.get("HTTP_CLIENT_PROXY")
    user_agent = os.environ.get("HTTP_CLIENT_USER_AGENT", "ragflow-http-client")

    # 延迟导入：函数真正被调用时才 import ssrf_guard，让本模块在未安装该
    # 依赖的环境里也能被 import。
    from common.ssrf_guard import assert_url_is_safe, pin_dns_global

    # --- 手动逐跳跟随重定向链，而不是用 httpx 的自动跟随 ---
    # 原因：每一跳的新地址都要先过一遍 SSRF 检查，自动跟随做不到。
    # 循环条件是 <=，所以最多发 6 次请求：最初的 1 次 + 最多 5 跳重定向。
    while redirect_hops <= _OAUTH_AVATAR_MAX_REDIRECTS:
        # 安全检查①：先解析域名，确认解析出来的 IP 是公网可路由地址
        # （127.0.0.1、10.x、169.254.x 这类内网/特殊地址会被拒绝）。
        # 返回 hostname = 域名，pin_ip = 刚刚校验过的那个公网 IP。
        try:
            hostname, pin_ip = assert_url_is_safe(current_url)
        except ValueError as exc:
            logger.warning("download_img rejected URL (SSRF guard): %s", exc)
            return ""

        import httpx

        timeout = httpx.Timeout(request_timeout)
        headers = {}
        if user_agent:
            headers["User-Agent"] = user_agent

        async def _stream_one_get() -> tuple[str, str | None]:
            """发出「一次」请求，返回三种结果之一：
            ('redirect', 新URL)   → 服务器让我们去别处取；
            ('data', data URI)    → 图片拿到了；
            ('fail', None)        → 失败（状态码不对/缺 Location 头/体积超限）。
            """
            # 安全检查②：DNS 锁定。把域名钉死在刚才校验过的那个公网 IP 上，
            # 防止「DNS 重绑定」攻击——校验时域名解析到公网 IP，真正发请求时
            # 又偷偷解析到内网 IP。pin_dns_global 是 with 上下文管理器，
            # 代码块结束自动解锁。
            with pin_dns_global(hostname, pin_ip):
                # 语法：async with = 协程里用的 with。客户端的创建/销毁本身
                # 涉及异步 I/O，所以要 await。
                # follow_redirects=False：不让 httpx 自动跟随重定向——我们在
                # 外层 while 里手动跟随，才能每一跳都检查。
                async with httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=False,
                    proxy=proxy,
                ) as client:
                    # stream = 流式响应：不等整个响应体下载完就先拿到响应头，
                    # 下面逐块读内容，避免超大响应一口气撑爆内存。
                    async with client.stream("GET", current_url, headers=headers or None) as response:
                        # 情况一：301/302/303/307/308 → 重定向。不读响应体，
                        # 从 Location 响应头里取下一跳地址，交给外层循环再跑一跳。
                        if response.status_code in _REDIRECT_STATUS:
                            await response.aclose()
                            location = response.headers.get("location")
                            if not location:
                                logger.warning(
                                    "download_img redirect missing Location header: status=%s redirect_hops=%s",
                                    response.status_code,
                                    redirect_hops,
                                )
                                return ("fail", None)
                            # urljoin 处理相对地址：
                            # 基准 "https://a.com/x" + "/y" → "https://a.com/y"
                            return ("redirect", urljoin(current_url, location))
                        # 情况二：连 200 都不是 → 直接判失败
                        if response.status_code != 200:
                            logger.warning(
                                "download_img non-200 response: status=%s redirect_hops=%s",
                                response.status_code,
                                redirect_hops,
                            )
                            return ("fail", None)
                        # 情况三：200 → 逐块读取响应体。
                        # bytearray = 可变字节序列，可以不停往后追加的 bytes。
                        body = bytearray()
                        # 语法：async for = 每轮迭代都可能 await 网络；数据还
                        # 没到时协程让出执行权，事件循环先去处理别的请求。
                        async for chunk in response.aiter_bytes():
                            # 加上这一块会超限 → 立刻断开，防止恶意服务器用
                            # 无限数据撑爆我们的内存
                            if len(body) + len(chunk) > _OAUTH_AVATAR_MAX_BYTES:
                                logger.warning(
                                    # codeql[py/clear-text-logging-sensitive-data]
                                    # 并非泄密：本分支刻意不把 current_url 写进
                                    # 日志参数——OAuth 令牌可能嵌在 URL 的查询
                                    # 串里。只记录大小阈值本身。
                                    "download_img response exceeded max size: max_bytes=%s",
                                    _OAUTH_AVATAR_MAX_BYTES,
                                )
                                await response.aclose()
                                return ("fail", None)
                            body.extend(chunk)
                        # 拼装 data URI：data:<内容类型>;base64,<base64 内容>。
                        # Content-Type 缺失时按 image/jpeg 兜底。
                        content_type = response.headers.get("Content-Type", "image/jpeg")
                        data_uri = "data:" + content_type + ";base64," + base64.b64encode(bytes(body)).decode("utf-8")
                        return ("data", data_uri)

        # 给刚定义的这个协程再套一层「总超时」：任何一环卡住超过
        # request_timeout 秒，直接取消（wait_for 超时抛 TimeoutError）。
        # 语法提示：_stream_one_get 定义在函数内部，并且直接使用了外层的
        # current_url、hostname 等变量——这叫闭包：内层函数记得外层作用域的值。
        try:
            kind, payload = await asyncio.wait_for(_stream_one_get(), timeout=request_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "download_img total wall-clock timeout: redirect_hops=%s timeout=%s",
                redirect_hops,
                request_timeout,
            )
            return ""
        except Exception as exc:
            logger.warning(
                "download_img request failed: redirect_hops=%s err=%s",
                redirect_hops,
                exc,
            )
            return ""

        # 按这一跳的结果分流：
        if kind == "redirect":
            # 服务器让我们换地址：记下新地址、跳数 +1，回到 while 顶部
            # 重新做安全检查再请求
            current_url = str(payload)
            redirect_hops += 1
            continue
        if kind == "fail":
            return ""
        # kind == "data"：拿到 data URI，返回给调用方
        return str(payload)

    # 能走到这里 = 重定向跳数用完了还没拿到图片（超过 5 跳还在跳转，
    # 八成是重定向死循环），放弃。
    # codeql[py/clear-text-logging-sensitive-data]
    # 并非泄密：本分支刻意不把 current_url 写进日志参数（防止嵌在 URL 里的
    # OAuth 令牌进日志）。只记录跳数和上限值。
    logger.warning(
        "download_img redirect hop limit exceeded: redirect_hops=%s max_redirects=%s",
        redirect_hops,
        _OAUTH_AVATAR_MAX_REDIRECTS,
    )
    return ""


def hash_str2int(line: str, mod: int = 10**8) -> int:
    """把字符串哈希成一个稳定的整数：同样的输入，任何时候都得到同样的输出。

    参数：
        line：任意字符串，例如一个切片 ID "chunk-abc-123"。
        mod：结果的取值上界（不含），默认 10**8（= 100000000，10 的 8 次方）。
            语法：a**b 表示 a 的 b 次方。

    返回值长这样：
        [0, mod) 区间内的整数，例如 hash_str2int("abc", 500) → 17（实测值）。
        实现：先算字符串的 SHA1 指纹（一个很大的十六进制数），转成整数后
        对 mod 取余，压进 [0, mod) 区间。
    """
    return int(hashlib.sha1(line.encode("utf-8")).hexdigest(), 16) % mod


def convert_bytes(size_in_bytes: int) -> str:
    """把字节数转成人类可读的文本，用于展示 ES 集群容量（调用方见
    doc_store 的 es_conn_base）。

    参数：
        size_in_bytes：字节数，例如 2048。

    返回值长这样：
        "2.00 KB"、"150 B"、"1.50 MB"——带单位的字符串。
        精度规则：数值 ≥100 不带小数；≥10 一位小数；<10 两位小数；
        单位是 B（字节）时永远是整数。
    """
    if size_in_bytes == 0:
        return "0 B"

    # 单位阶梯：从 B 开始，每满 1024 就升一级，直到数值 <1024 或升到 PB 为止
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0  # 当前停在第几级单位
    size = float(size_in_bytes)

    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1

    # 按数值大小选小数位数：数值越大，越不需要小数来保证精度
    if i == 0 or size >= 100:
        # f"{...:.0f}" = f-string 格式化（语法）：保留 0 位小数。
        # i == 0 时单位是 B，字节数本来就是整数，也不留小数。
        return f"{size:.0f} {units[i]}"
    elif size >= 10:
        return f"{size:.1f} {units[i]}"
    else:
        return f"{size:.2f} {units[i]}"


def once(func):
    """装饰器：让被装饰的函数「在本进程内只真正执行一次」，之后每次调用都
    直接返回第一次的结果。多线程安全——用锁保护，防止两个线程同时以为
    「自己是第一个」而执行两遍。

    参数：
        func：被装饰的函数，例如下面的 pip_install_torch。

    返回值长这样：
        一个与原函数同形的包装函数。第一次调用时执行 func 并缓存结果；
        此后每次调用都直接返回缓存值。
        用法（@ 是装饰器语法，见文件头小抄）：
            @once
            def f():
                return 42
            f()  # 第一次：真正执行，返回 42
            f()  # 之后：不再执行，直接返回缓存的 42

    小陷阱：如果 func 执行时抛了异常，executed 不会被置 True，
    下次调用会重试（这正是安装类函数想要的行为）。
    """
    # 下面三个变量活在闭包里（语法见小抄）：wrapper 每次被调用，
    # 读写的都是这同一份状态。
    executed = False  # 是否已经成功执行过一次
    result = None  # 第一次执行的结果缓存
    lock = threading.Lock()  # 互斥锁：同一时刻只允许一个线程进入 with 块

    def wrapper(*args, **kwargs):
        # nonlocal：声明「我要改的 executed/result 是外层 once 里的那两个」，
        # 不加这行，赋值会被当成创建 wrapper 自己的局部变量。
        nonlocal executed, result
        # with lock：进入时加锁、离开时自动解锁。两个线程同时到达时，
        # 后到的会在 with 这一行排队，等先到的执行完；等它进来时 executed
        # 已是 True，直接拿缓存结果。
        with lock:
            if not executed:
                result = func(*args, **kwargs)
                executed = True
        return result

    return wrapper


@once
def pip_install_torch():
    """用 pip 安装 PyTorch。只在 DEVICE 环境变量不是 cpu（即要用 GPU）时装；
    进程启动阶段由 common/settings.py 的 check_and_install_torch 调用。
    上面的 @once 保证多线程调用也只安装一次。

    无参数。返回值：无（CPU 模式直接返回，None 也会被 @once 缓存，
    之后再调用什么都不做）。
    """
    device = os.getenv("DEVICE", "cpu")
    if device == "cpu":
        # CPU 模式用不上 torch，直接跳过
        return
    logging.info("Installing pytorch")
    pkg_names = ["torch>=2.5.0,<3.0.0"]
    # subprocess.check_call = 起一个「子进程」执行命令（这是本文件里唯一
    # 真正创建新进程的地方！）：用当前 Python 解释器（sys.executable）跑
    # `pip install`，确保装进当前环境。这一行会阻塞到安装结束——发生在
    # 进程启动阶段，此时还没有用户请求，阻塞没关系。命令失败会抛异常。
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkg_names])


async def thread_pool_exec(func, *args, **kwargs):
    """把一个「同步阻塞函数」丢给一条临时工作线程去跑，协程 await 它的结果。
    等待期间事件循环不被卡住，可以继续处理别的 HTTP 请求。

    参数：
        func：要跑的同步函数（必须是普通 def 定义的，不能是 async def），
            例如嵌入模型的 encode、某个同步查库函数。
        *args：要传给 func 的位置参数，原样转发。
            例：thread_pool_exec(f, 1, 2) → 工作线程里执行 f(1, 2)。
        **kwargs：要传给 func 的关键字参数，原样转发。
            例：thread_pool_exec(f, k=3) → 工作线程里执行 f(k=3)。

    返回值：
        func 的返回值。例如 func 返回 {"ok": True}，await 本函数就拿到
        这个 dict。

    用法示例（在某个 async 的 HTTP 请求处理函数里）：
        vectors = await thread_pool_exec(embd_mdl.encode_queries, "你好")
    """
    # 拿到「当前正在运行本协程」的事件循环——也就是服务器进程里那位总调度。
    # 后面要把活儿递回给它安排线程。
    loop = asyncio.get_running_loop()
    # contextvars 快照：当前请求上下文里的变量（比如链路追踪 ID）只在本协程
    # 可见，工作线程是另一条线程，默认看不见；run_in_executor 又不会自动带
    # 过去（asyncio.to_thread 才会自动复制）。所以这里手动拍一张上下文快照，
    # 让工作线程用 ctx.run 在快照里执行函数，上下文就接上了。
    ctx = contextvars.copy_context()
    # 为本次调用新建一个只有 1 条线程的临时线程池。
    # 为什么每次都新建、不复用全局池：本环境（Python 3.13）下反复 await 共享
    # 池时可能触发死锁，一次性池子没有这个问题。
    # ⚠️ 注意 with 的副作用：离开 with 块会自动调用 shutdown(wait=True)，
    # 也就是「工作线程没跑完就一直等」。所以本函数只适合很快就干完的阻塞活儿；
    # 「可能跑好几分钟」的任务必须用下面的 thread_pool_exec_long_time，
    # 否则请求一旦中途被取消，退出 with 时还是会被拖住等到底。
    with ThreadPoolExecutor(max_workers=1) as executor:
        if kwargs:
            # functools.partial：把 func 和全部参数预先打包成一个「新函数」，
            # 因为 run_in_executor 转发关键字参数不方便，这里先捆好再递进去。
            inner = functools.partial(func, *args, **kwargs)
            # run_in_executor(池, 函数, *参数) = 让池里的工作线程执行
            # 「函数(参数)」。这里工作线程实际执行的是 ctx.run(inner)，
            # 即「在上下文快照 ctx 里运行 inner」。
            # await 期间：当前协程挂起，事件循环去处理别的请求；工作线程
            # 干完活，事件循环再唤醒本协程，拿到返回值。
            return await loop.run_in_executor(executor, ctx.run, inner)
        # 没有关键字参数时不用打包，直接把 func 和位置参数递进去
        return await loop.run_in_executor(executor, ctx.run, func, *args)


# 本函数与上面 thread_pool_exec 的唯一区别是「用哪个线程池」，而这决定了
# 请求中途被取消时的行为，正好回答「为什么需要两个版本」：
#
#   场景：用户发起「删除知识库」，里面有几个 GB 的文件，同步删除要跑几分钟。
#
#   如果用 thread_pool_exec（临时池）：客户端等不及断开连接 → Quart 取消这个
#   请求协程 → 协程从 await 处被唤醒并抛出取消异常 → 代码退出 with 块 →
#   with 的清理动作 shutdown(wait=True) 会等工作线程跑完 → 删除还是卡着
#   把取消流程拖住，等于白取消。
#
#   用本函数（全局池）：请求被取消时，没有任何人去 shutdown 这个全局池 →
#   工作线程继续把删除跑完（这正是我们想要的：删除必须干完，否则留下
#   垃圾数据），而事件循环立刻腾出手继续伺候别的请求。
#   代价：Python 无法强杀一个正在跑的工作线程，任务一旦开工就会跑到底。
async def thread_pool_exec_long_time(func, *args, **kwargs):
    """把「可能跑很久的同步阻塞函数」丢给全局长命工作池（本文件开头的
    _LONG_TIME_THREAD_POOL_EXECUTOR），协程 await 它的结果。

    参数与返回值：和 thread_pool_exec 完全一样（同步函数 + 原样转发的
    *args/**kwargs，返回 func 的返回值）。

    什么时候选它：任务耗时可能超过一次 HTTP 请求的耐心（大批量删除文档、
    删除知识库这类清理工作）。目前调用方就是这两类清理路径
    （document_api 的 FileService.delete_docs、dataset_api_service 的
    _delete_datasets_sync）。
    """
    loop = asyncio.get_running_loop()
    # 与 thread_pool_exec 同理：拍一张上下文快照带进工作线程
    ctx = contextvars.copy_context()
    if kwargs:
        inner = functools.partial(func, *args, **kwargs)
        # 递进全局池。池默认只有 1 条线程：同时来两个大删除任务时，
        # 第二个会在池内部排队，等第一个跑完——不会无限开线程，
        # 也不会跟事件循环自己的默认线程池抢资源。
        return await loop.run_in_executor(_LONG_TIME_THREAD_POOL_EXECUTOR, ctx.run, inner)
    return await loop.run_in_executor(_LONG_TIME_THREAD_POOL_EXECUTOR, ctx.run, func, *args)


class _CanonKey:
    """「规范化键」的防碰撞包装：把 _canonicalize 产出的哈希结构包一层，
    防止它与 dict/set 里某个不相关的普通值撞上（比如某个字符串恰好等于
    另一个值的 repr() 文本）。

    语法：__slots__ 声明本类实例只允许有 _key 这一个属性（省内存、禁止乱加）；
    __eq__ 定义「两个对象何时算相等」，__hash__ 定义「怎么算哈希码」。
    约定：相等的对象哈希码必须相等（dict/set 先按哈希码定位桶，再用
    __eq__ 精确比对），下面两个方法正是成对实现的。
    """

    __slots__ = ("_key",)

    def __init__(self, key):
        self._key = key

    def __eq__(self, other):
        # 只跟同类包装比：内部的规范化键相等，两个包装才算相等
        return isinstance(other, _CanonKey) and self._key == other._key

    def __hash__(self):
        # 直接用内部键的哈希码
        return hash(self._key)


def _canonicalize(value):
    """递归地把「不可哈希」的值（dict / list / set）转换成等价的哈希形态，
    并且保持相等关系不变：转换前相等的两个值，转换后依然相等。

    背景知识：Python 里只有「不可变」的对象才能当 dict 键 / set 元素
    （这种性质叫「可哈希」）。dict / list / set 是可变的、不可哈希的，
    直接当键会抛 TypeError。

    各类型的等价规则（为什么这么转）：
        - dict：相等不看键的顺序（{"a":1,"b":2} == {"b":2,"a":1}）
          → 用 frozenset 存「键-值对」（frozenset 天然无序，顺序不影响相等）；
        - list：相等看顺序 → 用 tuple（有序）存；
        - tuple：单独打标，因为在 Python 里 list 和 tuple 永不相等
          （[1] != (1,)），混用同一个标会出错；
        - set / frozenset：两者可以相等（{1} == frozenset({1})），共用一个标；
        - 其他连哈希都过不了的怪东西：退而存它的 repr()（打印出来的文本）。

    参数：
        value：任意 JSON 风格的值，例如 {"name": "x", "tags": ["a", "b"]}。

    返回值长这样（上面例子的转换结果，结构真实、便于想象）：
        ("__dict__", frozenset({
            ("name", "x"),
            ("tags", ("__list__", ("a", "b"))),
        }))
    """
    if isinstance(value, dict):
        # 字典 → （"__dict__" 标, 键值对组成的 frozenset）。frozenset 无序，
        # 正好匹配「字典相等不看键序」的规则。值递归规范化。
        return ("__dict__", frozenset((k, _canonicalize(v)) for k, v in value.items()))
    if isinstance(value, list):
        # 列表 → （"__list__" 标, 有序 tuple）。tuple 保序，匹配列表的相等规则
        return ("__list__", tuple(_canonicalize(v) for v in value))
    if isinstance(value, tuple):
        # 元组 → 单独的 "__tuple__" 标，和 list 区分开（[1] != (1,)）
        return ("__tuple__", tuple(_canonicalize(v) for v in value))
    if isinstance(value, (set, frozenset)):
        # 集合 → （"__set__" 标, frozenset）。set 与 frozenset 共用一个标，
        # 因为 {1} == frozenset({1})
        return ("__set__", frozenset(_canonicalize(v) for v in value))
    try:
        # 到这里的是「原子值」：字符串、数字、None 等。先试一下能不能哈希，
        # 能哈希就原样返回（不包装，开销为零）
        hash(value)
        return value
    except TypeError:
        # 连哈希都过不了的极端情况：存它的文本表示兜底
        return ("__repr__", repr(value))


def hashable_key(value):
    """把任意值变成「能当 dict 键 / set 元素」的东西。可哈希的普通值
    （字符串、数字、tuple 等）原样返回、零开销；只有不可哈希的值
    （dict、list 等）才走 _canonicalize 兜底转换。

    参数：
        value：任意值。例如 "doc-123"（可哈希，原样返回），
        或 {"id": "doc-123"}（不可哈希，返回 _CanonKey 包装）。

    返回值长这样：
        要么 value 本身，要么一个 _CanonKey 包装对象。调用方直接拿它当键用：
            seen = {}
            seen[hashable_key(item)] = item   # 无论 item 长什么样都不会抛异常
    目前用于 task_executor 侧 dataset_structure_merger 的去重。

    细节：顶层的 frozenset 会原样返回（不转成 set 的形态），因为它本身
    可哈希；只有当 frozenset 嵌在某个不可哈希值「内部」时才会被规范化。
    去重的实际内容（ID、描述文本）从不以集合形式出现，所以这条细节
    在热路径上没有任何开销。
    """
    try:
        # 先试哈希：大多数值（字符串 ID 等）在这里就直接返回了
        hash(value)
        return value
    except TypeError:
        # 不可哈希：递归规范化后包一层 _CanonKey，让它能安全地当键
        return _CanonKey(_canonicalize(value))
