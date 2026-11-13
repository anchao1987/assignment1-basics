import json  # 用于读取和解析存储词表与合并规则的 JSON 文件
from collections import deque  # 引入双端队列，用于在流式编码中维护固定数量的末尾位置
from collections.abc import Iterable, Iterator  # 引入类型提示相关的抽象基类
from pathlib import Path  # 用于跨平台的面向对象路径操作

import regex as re  # 引入第三方 regex 库（比标准库 re 支持更复杂的正则特性，如前瞻/后顾断言）

# 从自定义 BPE 模块导入预分词正则模式 PAT 以及合并 pretoken 的辅助函数 _merge_pretoken
from cs336_basics.bpe import PAT, _merge_pretoken


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """初始化 Tokenizer 类

        :param vocab: 词表字典，映射 token_id (int) 到 byte 序列 (bytes)
        :param merges: BPE 合并规则列表，元组形式 ((bytes, bytes))
        :param special_tokens: 特殊 token 字符串列表 (如 ["<|endoftext|>"])
        """
        self.vocab = vocab.copy()  # 浅拷贝传入的词表，防止外部修改影响内部状态
        self.merges = list(merges)  # 转换合并规则为列表存储

        if special_tokens:
            # 过滤重复的特殊 token，同时保持原列表中的出现顺序
            self.special_tokens = list(dict.fromkeys(special_tokens))
        else:
            self.special_tokens = []  # 未传入时初始化为空列表

        if "" in self.special_tokens:
            # 特殊 token 不能包含空字符串，否则会导致正则匹配陷入死循环
            raise ValueError("Special tokens must be nonempty strings")

        # 将特殊 token 补充注册到词表中
        for special_token in self.special_tokens:
            # 若该特殊 token 的 UTF-8 字节流已在词表值中，跳过
            if special_token.encode("utf-8") in self.vocab.values():
                continue
            else:
                # 否则给它分配一个新的 ID（取当前最大 ID + 1），并写入词表
                self.vocab[max(self.vocab, default=-1) + 1] = special_token.encode(
                    "utf-8"
                )

        # 构建反向词表映射：从字节流 (bytes) 到 Token ID (int)
        self.reverse_vocab = {}
        for token_id, token_bytes in self.vocab.items():
            self.reverse_vocab[token_bytes] = token_id

        # 构建 BPE 合并优先级字典：(left_bytes, right_bytes) -> 合并优先级 rank (越小优先级越高)
        self.merge_ranks = {}
        for rank, pair in enumerate(self.merges):
            self.merge_ranks[pair] = rank

        # 按长度降序排列特殊 token（优先匹配更长的特殊 token，防止长 token 被短 token 切碎）
        ordered_specials = sorted(self.special_tokens, key=len, reverse=True)
        # 将特殊 token 编译为正则是转义过的正则表达式；若没有特殊 token 则为 None
        self._special_pattern = (
            re.compile("|".join(re.escape(token) for token in ordered_specials))
            if ordered_specials else None
        )
        # 编译通用预分词（Pre-tokenization）正则表达式
        self._pretoken_pattern = re.compile(PAT)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """类方法：从存储 JSON 格式的文件中加载词表和合并规则并实例化 Tokenizer

        例如 vocab 格式: {"256": [97, 98]} -> 转换为 {256: b'ab'}
        merges 格式: [[[97], [98]]] -> 转换为 [(b'a', b'b')]
        """
        # 读取词表文件，将 key 转为 int，字节数组序列 [97, 98] 转为 bytes 对象
        with Path(vocab_filepath).open(encoding="utf-8") as file:
            vocab = {int(token_id): bytes(values) for token_id, values in json.load(file).items()}

        # 读取合并规则文件，将字节列表序列转换为 bytes 元组
        with Path(merges_filepath).open(encoding="utf-8") as file:
            merges = [(bytes(left), bytes(right)) for left, right in json.load(file)]

        # 调用构造函数完成初始化
        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int]) -> str:
        """解码：将 Token ID 列表转换为原始文本字符串"""
        byte_parts = []
        for token_id in ids:
            # 从词表中找到对应 ID 的字节序列并收集
            byte_parts.append(self.vocab[token_id])
        # 将所有字节片段拼接成完整字节流
        data = b"".join(byte_parts)
        # 将字节流解码为 UTF-8 字符串；非法 UTF-8 字节会被替换为官方替换字符 (U+FFFD)
        return data.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        """编码：将输入文本转为 Token ID 列表"""
        # 遍历文本切割后的每个片段 (piece)，并对其进行 BPE 编码，最后平铺展开成一个 ID 列表
        return [token_id for _, piece in self._iter_pieces(text) for token_id in self._encode_piece(piece)]

    def _iter_pieces(self, text: str) -> Iterator[tuple[int, str]]:
        """生成器：按顺序切分文本。匹配特殊 token 并保持其完整，其余文本使用预分词正则切分。"""
        start = 0
        if self._special_pattern is not None:
            # 扫描文本中的所有特殊 token
            for special in self._special_pattern.finditer(text):
                # 对特殊 token 之前的普通文本，使用预分词正则切片
                for match in self._pretoken_pattern.finditer(text, start, special.start()):
                    yield match.start(), match.group()
                # 产出特殊 token 本身（起始位置和完整文本）
                yield special.start(), special.group()
                # 更新下一个扫描起始位置
                start = special.end()
        # 对剩余的文本（或没有特殊 token 时）进行最后一次预分词切片
        for match in self._pretoken_pattern.finditer(text, start):
            yield match.start(), match.group()

    def _encode_piece(self, piece: str) -> list[int]:
        """对单条片段 (piece) 执行 BPE 算法合并并映射为 ID 列表"""
        # 如果该片段是特殊 token，直接返回其对应的单 ID
        if piece in self.special_tokens:
            return [self.reverse_vocab[piece.encode("utf-8")]]

        # 将片段转换为 UTF-8 字节，初始每个字节作为一个独立的单字节片段
        pretoken = tuple(bytes([value]) for value in piece.encode("utf-8"))

        # 循环迭代合并相邻字节对，直到不能再合并或长度只剩 1
        while len(pretoken) > 1:
            best_rank = float("inf")  # 初始化最高优先级（最优先的 rank 最小）
            best_pair = None  # 记录当前最高优先级的相邻对

            # 遍历所有相邻对 (pair)
            for pair in zip(pretoken, pretoken[1:]):
                # 查询该对在合并表中的 rank，找不到则为 inf
                rank = self.merge_ranks.get(pair, float("inf"))
                if rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            # 如果所有的相邻对都不在合并规则表中，说明达到了最高合并程度，退出循环
            if best_pair is None:
                break

            # 执行合并：将序列中的 best_pair 替换为合并后的字节片段
            pretoken = _merge_pretoken(pretoken, best_pair, b"".join(best_pair))

        # 从反向词表中查找合并后的各个字节片段对应的 Token ID 并返回
        return [self.reverse_vocab[token] for token in pretoken]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性流式编码：对输入的文本流逐块 (chunk) 进行处理，安全地处理跨块可能被截断的 Token。"""
        pending = ""  # 暂存区：保存未完成分词的尾部字符串
        # 获取特殊 token 中的最大字符长度
        special_length = max(map(len, self.special_tokens), default=0)

        for chunk in iterable:
            pending += chunk  # 将新到的块追加到暂存区
            trailing_starts = deque(maxlen=2)  # 维持最后两个分词片段的起始位置
            safe_start = 0  # 记录当前绝对安全（不受跨块截断影响）的切分起始点

            for start, _ in self._iter_pieces(pending):
                trailing_starts.append(start)  # 记录片段的起始偏移量
                # 如果当前起始偏移距离缓冲区末端超过了最长特殊 token 的长度，则说明前面的部分是安全的
                if start <= len(pending) - special_length:
                    safe_start = start

            # 确定安全截断边界 cutoff：
            # 必须同时满足 1. 不处于特殊 token 匹配风险区 (safe_start)
            # 2. 回退至少 2 个片段，保证预分词正则的前瞻 (Lookahead) 不被破坏
            cutoff = min(safe_start, trailing_starts[0]) if trailing_starts else 0

            if cutoff:
                # 编码并产出 (yield) [0:cutoff] 安全区域的字符
                yield from self.encode(pending[:cutoff])
                # 剩余未安全编码的部分保留在 pending 中供下一轮处理
                pending = pending[cutoff:]

        # 文本流全部结束后，编码并产出最后残留的所有 pending 内容
        yield from self.encode(pending)