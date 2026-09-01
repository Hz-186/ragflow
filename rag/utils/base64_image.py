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

import base64
import logging
from functools import partial
from io import BytesIO

from PIL import Image


from common.misc_utils import thread_pool_exec
from rag.utils.lazy_image import open_image_for_processing

test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAAA6ElEQVR4nO3QwQ3AIBDAsIP9d25XIC+EZE8QZc18w5l9O+AlZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBWYFZgVmBT+IYAHHLHkdEgAAAABJRU5ErkJggg=="
test_image = base64.b64decode(test_image_base64)


async def image2id(d: dict, storage_put_func: partial, objname: str, bucket: str = "imagetemps"):
    """把切片上的图片对象「存进对象存储、换成一个字符串引用」—— 图片入库换票员。

    输入数据的样子：
        d —— 切片文档 {"id": "a1b2...", "content_with_weight": "...", "image": <PIL 图>}
        objname —— 存储对象名（主路径 task_executor 传切片 id，图与切片同名同命；
                   另有少量调用方传随机 UUID）
        bucket —— 存储桶名（主路径传知识库 id；本函数默认值 "imagetemps"）
    输出（原地修改 d）：
        image 字段被弹出（PIL 对象没法进 ES），换成
        d["img_id"] = "桶名-对象名"（检索端展示图片时凭它回读）

    干了三件事：
        ① 把图片物化后编码成 JPEG 二进制（RGBA/P 模式先转 RGB，因为 JPEG 不支持透明通道）
        ② 经限流器把二进制存进对象存储（MinIO 等）
        ③ 在切片上写 img_id 引用，image 字段从此消失
    """
    import logging
    from io import BytesIO
    from rag.svr.task_executor_limiter import minio_limiter

    if "image" not in d:
        return
    if not d["image"]:
        del d["image"]
        return

    image = d.pop("image")  # 先把图片从切片上弹出（处理完只在切片上留 img_id 引用）

    def encode_image():
        # ① 把图片（可能是 PIL 图 / LazyImage）物化成可编码的图像对象
        img, close_after = open_image_for_processing(image, allow_bytes=False)

        if isinstance(img, bytes):
            return bytes(img)

        if not isinstance(img, Image.Image):
            return None

        owned_images = [img] if close_after else []
        try:
            img.load()
            if img.mode in ("RGBA", "P"):
                converted = img.convert("RGB")  # JPEG 不支持透明/调色板模式，先转 RGB
                owned_images.append(converted)
                img = converted

            with BytesIO() as buf:
                img.save(buf, format="JPEG")  # ① 统一编码成 JPEG
                return buf.getvalue()
        except (OSError, ValueError) as e:
            logging.warning(f"Saving image exception: {e}")
            return None
        finally:
            for owned_img in owned_images:
                try:
                    owned_img.close()
                except Exception:
                    pass

    jpeg_binary = await thread_pool_exec(encode_image)  # 编码是 CPU 密集活，丢线程池避免卡事件循环
    if jpeg_binary is None:
        return

    async with minio_limiter:
        # ② 经限流器上传对象存储（防止并发写爆 MinIO 连接）
        await thread_pool_exec(lambda: storage_put_func(bucket=bucket, fnm=objname, binary=jpeg_binary))

    d["img_id"] = f"{bucket}-{objname}"  # ③ 写回「桶名-对象名」复合引用


def parse_storage_composite_id(composite_id: str) -> tuple[str, str] | None:
    """把 ``{桶名}-{对象名}`` 复合 id 拆回 (桶名, 对象名) —— img_id 解码器。

    只在第一个连字符处切分：对象名里可能还带连字符（如 ``page-1.jpg``）。
    ``image2id`` 存入的 img_id 形如 ``f"{bucket}-{objname}"``。

    Args:
        composite_id: Composite storage identifier.

    Returns:
        ``(bucket, object_key)`` when valid, otherwise ``None``.
    """
    parts = composite_id.split("-", 1)
    if len(parts) != 2 or not parts[0] or not parts[1] or composite_id.endswith("-"):
        return None
    return parts[0], parts[1]


def id2image(image_id: str | None, storage_get_func: partial):
    """凭 img_id 从对象存储里「换回」图片 —— 图片出库取件员（image2id 的逆操作）。

    检索命中后前端要展示切片图：用 img_id 拆出桶名/对象名，读回字节流再打开成 PIL 图。
    id 不合法或读取失败时返回 None。

    Args:
        image_id: Value produced by ``image2id`` (``{bucket}-{object_key}``).
        storage_get_func: Callable ``(bucket=, fnm=)`` returning raw bytes.

    Returns:
        A PIL ``Image`` instance, or ``None`` when the ID is invalid or load fails.
    """
    if not image_id:
        return
    parsed = parse_storage_composite_id(image_id)
    if not parsed:
        logging.debug("Invalid image_id composite format: %s", image_id)
        return
    bkt, nm = parsed
    try:
        blob = storage_get_func(bucket=bkt, fnm=nm)
        if not blob:
            return
        return Image.open(BytesIO(blob))
    except Exception as e:
        logging.exception(e)
