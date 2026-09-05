# -*- coding: utf-8 -*-
"""
====================================================================
 教学 demo：协程 vs 线程 vs thread_pool_exec —— 四个对照实验
====================================================================
运行方式（Git Bash，在仓库根目录）：
    PYTHONIOENCODING=utf-8 python study_demo_async_vs_thread.py
（Windows 若直接双击/在 cmd 里跑出现乱码报错，就加上前面那段
 PYTHONIOENCODING=utf-8，意思是"强制用 UTF-8 编码打印中文"。）

本文件只用 Python 标准库，不需要安装项目任何依赖。

四个实验回答四个问题：
  实验 0   调用 async def 函数 = 执行了吗？        → 没有！只是造了一张"待办卡"
  实验 1   协程并发需要几条线程？                  → 1 条（主线程），3 个任务各睡 1 秒总共只花 1 秒
  实验 2   在协程里裸调同步阻塞函数会怎样？        → 调度员被卡死，3 个任务排队花 3 秒，心跳骤停
  实验 3   thread_pool_exec 怎么救场？             → 阻塞的活丢给临时线程，又变回 1 秒，心跳全程正常

  实验 3 里的 thread_pool_exec 是 ragflow 真实实现
  （common/misc_utils.py:490）的逐行简化复刻，骨架完全一致。

什么时候用协程、什么时候用线程？记住一句话：
  * 对方是【异步库】写的（aiohttp、litellm 的 a 系列……）→ 直接 await，零线程；
  * 对方是【老式同步函数】（plain def，内部死等网络/磁盘）→ 绝不能裸调，
    必须用 thread_pool_exec 手动丢给线程。
  * 事件循环自己【永远不会】主动开线程——开线程是写代码的人的决定。
"""

import asyncio  # Python 官方的"协程 + 事件循环"工具箱
import functools  # 函数加工小工具；本 demo 用到 partial（把函数和参数预先捆成一包）
import threading  # 线程工具箱；本 demo 只用 current_thread() 看"我现在在哪条线程上"
import time  # 计时；以及 time.sleep（原地死等，是实验 2 的反面教材）
import contextvars  # "上下文变量"工具箱；本 demo 用 copy_context() 给当前上下文拍快照
from concurrent.futures import ThreadPoolExecutor  # 线程池：按需创建/管理工作线程的容器


def which_thread():
    """观测仪器：返回当前这行代码正在哪条线程上执行。

    返回值长这样：
        "MainThread"                  ← 主线程（事件循环就住在它上面）
        "ThreadPoolExecutor-0_0"      ← 第 0 个线程池的第 0 条工作线程
    后面每个实验都靠打印它来"亲眼看见"代码跑在谁身上。
    """
    return threading.current_thread().name


# ====================================================================
# 实验 0：调用协程函数 ≠ 执行
# ====================================================================
# async def = 定义一个"协程函数"。它与普通 def 最大的区别：
# 调用它不会执行函数体，只会返回一个"协程对象"——相当于一张写着
# 待办事项的卡片。卡片必须交给事件循环（await 它 / create_task 它）
# 才会真正被执行。
async def say_hello():
    print("    你好！")


def experiment_0():
    print("\n== 实验 0：调用协程函数 ≠ 执行 ==")
    card = say_hello()  # 注意：这一行没有打印"你好！"——函数体根本没跑
    print(f"  调用 say_hello() 得到的东西：{card}")
    #  ↑ 输出类似 <coroutine object say_hello at 0x000001A2B3C4D5E0>
    #    这就是那张"待办卡"（协程对象），不是执行结果。
    card.close()  # 不打算执行的卡片要 close() 扔掉，否则 Python 退出时会警告


# ====================================================================
# 实验 1：原生协程 —— 3 个任务并发，只用 1 条线程
# ====================================================================
async def async_io_task(name, seconds):
    """模拟一个用【异步库】写的任务。

    对应仓库里的真实例子：rag/llm/chat_model.py:1173 的 _post_json，
    用 aiohttp（异步 HTTP 库）调 LLM，全程 await，零线程。

    参数：
        name：任务名，例如 "任务1"
        seconds：要"睡"几秒，例如 1
    返回值：无（只打印）
    """
    print(f"  [{name}] 开始，我现在在：{which_thread()}")
    # await asyncio.sleep(秒) = 合作式睡觉：
    #   向事件循环登记"N 秒后叫醒我"，然后【立刻让位】，
    #   事件循环转身去跑别的协程。等待期间没有任何线程被占着。
    # ★ 这就是 await 的本质：不是"傻等"，是"挂起自己+让出调度权"。
    await asyncio.sleep(seconds)
    print(f"  [{name}] 结束，我现在在：{which_thread()}")


async def experiment_1():
    print("\n== 实验 1：原生协程 —— 3 个任务各'睡'1 秒 ==")
    t0 = time.perf_counter()  # 记下开始时刻（高精度秒表）

    # 下面一行是"列表推导式"：for i in (1,2,3) 循环三次，每次
    # 造一个协程对象并用 create_task 交给事件循环排期。
    # asyncio.create_task(协程) = "调度员，这张待办卡你收好，有空就跑它"
    #   ——注意它不等待，任务们从此【并发】推进。
    tasks = [asyncio.create_task(async_io_task(f"任务{i}", 1)) for i in (1, 2, 3)]

    # await asyncio.gather(*tasks) = 等这批任务【全部】完成再往下走。
    # * 号是"解包"：把列表 [t1,t2,t3] 摊开成三个独立参数 gather(t1,t2,t3)。
    await asyncio.gather(*tasks)

    print(f"  总耗时：{time.perf_counter() - t0:.2f} 秒  ← 约 1 秒，不是 3 秒！")
    print("  为什么？三个任务的'等待'是重叠的：任务1挂起时，任务2、3 也在推进。")
    print(f"  全程只有一条线程：{which_thread()}，事件循环没有开任何线程。")


# ====================================================================
# 实验 2：同步阻塞 —— 调度员被卡死（反面教材）
# ====================================================================
def blocking_work(name, seconds):
    """老式【同步】函数：内部死等，绝不让位。

    对应仓库里的真实例子：rag/app/naive.py 的 chunk（切 PDF）、
    api/db/services/dialog_service.py 的 DialogService.query（Peewee 查库）。

    time.sleep(秒) = 原地死等。谁调用它，谁就被冻结——
    如果调用者是协程（跑在主线程上），整个事件循环跟着一起冻结。

    参数：name=任务名（如 "任务1"）；seconds=死等几秒（如 1）
    返回值：字符串，如 "任务1 的结果"
    """
    print(f"    [{name}] 死等开始，我现在在：{which_thread()}")
    time.sleep(seconds)  # ★ 反面教材：没有 await，纯粹卡住当前线程
    print(f"    [{name}] 死等结束，我现在在：{which_thread()}")
    return f"{name} 的结果"


async def sync_task(name):
    """在协程里【裸调】同步阻塞函数——实验 2 的错误示范。"""
    print(f"  [{name}] 开始，我现在在：{which_thread()}")
    result = blocking_work(name, 1)  # ← 没有 await！主线程（=事件循环）被卡死 1 秒
    print(f"  [{name}] 结束，拿到 {result!r}")


async def heartbeat():
    """心跳协程：每 0.25 秒报一次"我还活着"。

    它是【调度员的示波器】：心跳正常跳 = 事件循环还在自由调度；
    心跳停了 = 事件循环被某个同步调用卡住了。永不返回（while True），
    由调用方负责 cancel（取消）。
    """
    while True:  # 无限循环——协程版死循环不怕，因为每次 sleep 都让位
        await asyncio.sleep(0.25)
        print("      [心跳] 事件循环还活着...")


async def experiment_2():
    print("\n== 实验 2：同步阻塞 —— 同样 3 个任务各 1 秒，但裸调 ==")
    t0 = time.perf_counter()

    hb = asyncio.create_task(heartbeat())  # 先开心跳，当示波器
    tasks = [asyncio.create_task(sync_task(f"任务{i}")) for i in (1, 2, 3)]
    await asyncio.gather(*tasks)

    print(f"  总耗时：{time.perf_counter() - t0:.2f} 秒  ← 3 秒！只能排队。")
    print("  数一数上面的心跳：3 秒里本应跳约 12 次，实际【一次都没跳】——")
    print("  time.sleep 死等时主线程被冻住，心跳协程完全得不到调度，")
    print("  等 3 个任务把循环解冻，实验已经结束、心跳被取消了。")
    print("  这就是'一个同步调用拖垮全部并发'。")

    hb.cancel()  # 给心跳协程发"取消"信号（在它的下一个 await 处生效）
    # 被取消的任务会抛 CancelledError；gather 加 return_exceptions=True
    # 表示"异常当普通结果收，别炸"。这行 await 是确保取消真正执行完。
    await asyncio.gather(hb, return_exceptions=True)


# ====================================================================
# 实验 3：thread_pool_exec —— 同步脏活丢给临时线程
# --------------------------------------------------------------------
# ragflow 真实实现 common/misc_utils.py:490 的逐行简化复刻，
# 骨架、顺序、每一行的作用与真品一致。
# ====================================================================
async def thread_pool_exec(func, *args, **kwargs):
    """把一个【同步阻塞函数】丢给一条临时工作线程去跑，协程 await 它的结果。

    参数：
        func：要跑的同步函数（必须是普通 def 定义的），例如 blocking_work
        *args：转发给 func 的位置参数。*args 语法 = "收集所有多余的位置
            参数打包成元组"。例：thread_pool_exec(f, 1, 2) → args=(1, 2)
        **kwargs：转发给 func 的关键字参数。**kwargs = "收集所有多余的
            关键字参数打包成字典"。例：thread_pool_exec(f, k=3) → kwargs={"k": 3}

    返回值：func 的返回值。例：func 返回 "任务1 的结果"，
        await 本函数就拿到这个字符串。
    """
    # 第 1 步：拿到"正在运行本协程"的事件循环（那位总调度员）本人。
    #   后面要把活儿递回给它，由它安排线程并在干完时唤醒本协程。
    loop = asyncio.get_running_loop()

    # 第 2 步：给当前协程的"上下文变量"拍一张快照（contextvars）。
    #   上下文里可能装着链路追踪 ID 之类只在当前请求可见的变量；
    #   工作线程是另一条线程，默认看不见它们。拍快照 → 让线程在
    #   快照里干活（下面的 ctx.run），上下文就接上了。
    ctx = contextvars.copy_context()

    # 第 3 步：现场造一个只有【1 条】工作线程的临时线程池。
    #   with ... as executor: 语法 = 进入时创建，离开 with 块时自动清理
    #   （清理动作是 shutdown(wait=True)：等工作线程干完才放行——
    #    所以真品注释提醒：只适合"很快干完"的活，跑几分钟的大活要用
    #    全局常驻池的 thread_pool_exec_long_time，misc_utils.py:553）。
    with ThreadPoolExecutor(max_workers=1) as executor:
        if kwargs:
            # 第 4 步（只有关键字参数才需要）：functools.partial 把
            #   "func + 全部参数"预先捆成一个【无参数】的新函数。
            #   为什么要捆：run_in_executor 只接受位置参数转发，
            #   没法直接带 name=xx 这种关键字参数，先打包绕过限制。
            inner = functools.partial(func, *args, **kwargs)
            # 第 5 步：run_in_executor(池,  callable, *参数)
            #   = "池里的工作线程，去执行 callable(参数...)"。
            #   这里让线程执行的是 ctx.run(inner)，即"在上下文快照 ctx
            #   里运行 inner"。
            #   await = 本协程在这一行挂起；事件循环立刻去调度别人；
            #   线程干完把返回值送回，事件循环再唤醒本协程。
            return await loop.run_in_executor(executor, ctx.run, inner)
        # 没有关键字参数就不用打包：func 和位置参数直接递给线程池，
        # 线程里执行的是 ctx.run(func, *args)
        return await loop.run_in_executor(executor, ctx.run, func, *args)


async def wrapped_task(name, seconds, use_kwargs=False):
    """实验 3 的任务：同样调 blocking_work，但用 thread_pool_exec 包住。

    use_kwargs=True 时改用关键字参数传参，专门走一次上面第 4 步的
    functools.partial 打包分支，两条路都体验一遍。
    """
    print(f"  [{name}] 协程部分，我现在在：{which_thread()}")
    if use_kwargs:
        # 关键字参数版：thread_pool_exec 内部会走 partial 打包那条分支
        result = await thread_pool_exec(blocking_work, name=name, seconds=seconds)
    else:
        # 纯位置参数版：func 和 args 直接递进线程池
        result = await thread_pool_exec(blocking_work, name, seconds)
    # await 回来了。注意打印线程名：干脏活的是工作线程，
    # 但"收货"的这行又回到了主线程——协程被事件循环唤醒接着跑。
    print(f"  [{name}] 协程收到结果 {result!r}，收货线程：{which_thread()}")


async def experiment_3():
    print("\n== 实验 3：thread_pool_exec —— 同样 3 个同步任务各 1 秒 ==")
    t0 = time.perf_counter()

    hb = asyncio.create_task(heartbeat())  # 同样的心跳示波器
    tasks = [
        asyncio.create_task(wrapped_task("任务1", 1)),                    # 位置参数分支
        asyncio.create_task(wrapped_task("任务2", 1, use_kwargs=True)),   # partial 打包分支
        asyncio.create_task(wrapped_task("任务3", 1)),                    # 位置参数分支
    ]
    await asyncio.gather(*tasks)

    print(f"  总耗时：{time.perf_counter() - t0:.2f} 秒  ← 又回到约 1 秒！")
    print("  这次心跳跳满了约 4 次/秒 × 1 秒：工作线程死等时，事件循环全程自由。")
    print("  对比线程名：blocking_work 跑在 ThreadPoolExecutor-0_0 / -1_0 / -2_0")
    print("  （3 次调用 = 3 个临时池各 1 条线程，用完即毁），")
    print("  而协程的'开始/收货'两行始终在 MainThread —— 这就是分工。")

    hb.cancel()
    await asyncio.gather(hb, return_exceptions=True)


# ====================================================================
# 主流程
# ====================================================================
async def main():
    """四个实验按顺序跑一遍（本身也是个协程，由 asyncio.run 驱动）。"""
    await experiment_1()
    await experiment_2()
    await experiment_3()


# if __name__ == "__main__": = "本文件被直接运行时才执行下面代码"
# （被别的文件 import 时不执行，Python 标准写法）
if __name__ == "__main__":
    experiment_0()  # 实验 0 不需要事件循环，直接在主线程跑
    # asyncio.run(协程) = 总开关：创建事件循环 → 把 main() 这张"待办卡"
    # 交给它跑到完成 → 关闭循环。每个用 asyncio 的程序都有且只有这么一个入口。
    asyncio.run(main())

    print("""
==================== 总结 ====================
1) 协程（async def + await）：一条线程内的"让位式"并发。
   等待 IO 时挂起自己、让事件循环去跑别人 —— 零额外线程。
   前提：链路上下全是异步库（aiohttp、asyncio.sleep……）。
2) 线程：不是事件循环自动开的，是【开发者】看到同步老代码时
   手动用 thread_pool_exec 包的。每次调用临时开 1 条，用完即毁。
3) 事件循环 = 唯一的调度员，住在主线程。它只认 await：
   你 await，它就调度别人；你裸调 time.sleep，它就陪葬。
==============================================""")
