# Chapter 2：Byte-Pair Encoding (BPE) Tokenizer 教学讲解

本文将本章内容整理为 2.1 到 2.5 的学习笔记，目标是从字符编码基础过渡到 BPE 训练实践。

---

## 2.1 Unicode 标准

### 字符和unicode码点的对应关系
Python 里两个基础函数：
- `ord(char)`：字符 -> Unicode 码点整数
- `chr(code_point)`：Unicode 码点整数 -> 字符

示例：
```python
ord("展")      # 23637
chr(23637)     # "展"
```

这说明字符可以抽象成数字编号（即unicode码点）来处理。

### `chr(0)` 与空字符（NULL）
```python
chr(0)
chr(0).__repr__()   # '\x00'
```

核心结论：
- `chr(0)` 对应 `U+0000`（NULL 字符）。
- `repr` 形式可见为 `\x00`，但直接打印`chr(0)`通常不可见。
- 它虽然“看不见”，但确实存在于字符串中，可能影响底层处理。

例如：
```python
"this is a test" + chr(0) + "string"
```

中间包含一个不可见控制字符。(输出为：`this is a teststring`)

#### QA：
(a) chr(0) 返回的是 Unicode 码点 U+0000，即空字符（NULL character）。
(b) 它的 __repr__() 表示为 '\x00'（可见的转义形式），而直接打印时不会显示任何可见字符，因为它是不可见的控制字符。
(c) 当该字符出现在文本中时，它通常不会显示任何可见内容，但仍然存在于字符串内部并可能影响某些底层字符串处理或系统行为。

---

## 2.2 Unicode 编码（UTF-8）
Unicode 负责“给字符编号”，UTF-8 负责“把编号存成字节”。

示意：

1. 字符：`"牛"`
2. unicode码点：`U+725B`
3. UTF-8 字节：`E7 89 9B`（3 个字节）
4. 磁盘实际保存：字节序列

### 编码与解码
```python
test_string = "hello! こんにちは!"
utf8_encoded = test_string.encode("utf-8")  #b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'

type(utf8_encoded) # bytes
# 要访问 Python bytes 对象的底层字节值，我们可以对其进行迭代（例如调用 list()）
list(utf8_encoded) # [104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175]

utf8_encoded.decode("utf-8") # 'hello! こんにちは!'
```

- `encode("utf-8")`：字符串 -> 字节
- `decode("utf-8")`：字节 -> 字符串

### 观察底层字节与长度差异
```python
len(test_string)   # 13
len(utf8_encoded)  # 23
```

含义：
- 一个字符不一定对应一个字节。
- UTF-8 是变长编码，ASCII 常见字符 1 字节，其他字符可能多字节。
- 通过将 Unicode 码点转换为字节序列（例如通过 UTF-8 编码），我们本质上是在将码点序列（范围在 0 到 154,997 的整数）转换为字节值序列（范围在 0 到 255 的整数）

### python中的bytes类型讲解
```python
s = "low"   # str类型
b = s.encode("utf-8")   # b'low'
b[0:1] == b'l'
b[0] == 108  # 是int类型，对应l的ASCII码

bytes[(108)] == b'l'    # b表示bytes类型
```

#### QA：
(a) UTF-8 更节省空间（尤其对以 ASCII 为主的文本是 1 字节表示），无字节序问题（不像 UTF-16 有 endian 问题），且与 ASCII 完全兼容、互联网标准广泛使用，因此更适合基于 byte 的 tokenizer 训练；相比之下 UTF-16/32 更占空间且会引入额外的零字节模式。
(b) 例如输入 "こんにちは".encode("utf-8") 会产生错误结果，因为该函数逐字节单独解码，而 UTF-8 的多字节字符必须作为整体解码，拆开会导致解码失败或错误字符。
(c) 例如字节序列 b'\xff\xff' 无法解码为任何合法的 UTF-8 字符，因为 0xFF 在 UTF-8 中不是合法的起始或续字节。


---

## 2.3 子词分词（Subword Tokenization）

### 为什么不只用字节级分词

字节级分词能缓解词级分词的 OOV（词汇表外）问题，但代价是序列更长，训练/推理更慢。

### 子词分词的核心思路
- 纯字节词表只有 256 个条目（`0~255`）。
- 子词分词通过扩大词表来压缩常见模式。
- 高频序列（如 `b'the'`）可作为单个 token，减少序列长度。

### BPE 在做什么

BPE（Byte-Pair Encoding）通过不断合并高频相邻单元来提升压缩效率：

- 高频模式合并成新 token
- 未登录词可回退到字节级表示
- 在“泛化能力”和“序列长度”间取得平衡

---

## 2.4 BPE 分词器训练
有三个主要步骤：1. 词汇表初始化 2. 预分词 3. 计算 BPE 合并
1. 读原始文本（Unicode str  而不是bytes）
2. 用 special token 做切分
3. 对每段普通文本做 regex 预分词，得到一串 pre-token 字符串 
4. 把每个 pre-token 字符串 encode 成 UTF-8 bytes，并转成“bytes 序列/tuple”形式用于统计 
5. 统计每个 pre-token 的出现次数（计数表）
6. 后续 merge：只在每个 pre-token 内部统计 pair、合并 pair（禁止跨 pre-token）。

### 1. 词汇表初始化

字节级 BPE 的初始词表为全部单字节 token，共 256 项（`0~255`）。
**数据类型：** vocab: dict[int, bytes]（id → token_bytes）

### 2. 预分词
**原因：** 如果你直接在整段原始字节流上做统计与合并，每做一次 merge 都要全语料扫一遍，极慢，而且还会产生一些“语义上很接近但形式不同”的碎片 token（例如 dog! vs dog.）。
**预分词：** 先用一个较粗粒度的规则，把文本切成一段段 `pre-token`；之后的 BPE 统计/合并都只在每个 pre-token 内部进行。为了效率，BPE 训练时 不考虑跨 pre-token 边界的 pair。(且空格、标点形成了边界)

原始 BPE 实现仅通过空格分割进行预分词（即 s.split(" ")）。相比之下，我们将采用基于正则表达式的预分词器

```python
# 基于正则表达式的预分词器示例
import regex as re
# PAT即为预分词的正则表达式，涵盖了多种语言和符号的情况
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re.findall(PAT, "some text that i'll pre-tokenize")
# 预分词的结果：['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```
注意点：

- 空格常与后续词绑定（如 `' text'`）。
- 缩写可能被拆分（如 `i` 和 `'ll`）。
- 工程上更建议 `re.finditer` 流式遍历，边扫描边更新计数，降低内存。
- 需要最后的输出：`dict[tuple[bytes], int]` 其中：键是一个“token 序列”（初始时是单字节拆开的序列，例如 {(b'l', b'o', b'w'): 5 …}），值是出现次数。  

**需要注意的是，** 在 Python 中，即使是单个字节也是一个 bytes 对象。Python 中没有 byte 类型来表示单个字节。 
- 例子1：用x = b'a'来表示单个字节'a'，它是一个 bytes 对象，长度为 1。
- 例子2：data = b'hello' | data[0]的结果是104 (bytes是一个序列类型，访问单个元素得到的是 int)

### 3. 统计 BPE merge 单次迭代
拿到预分词输出`dict[tuple[bytes], int]`后，就可以开始统计 BPE 合并单元了。
- 每做 1 次 merge，你就新增 1 个 token，所以最终 vocab 大小 = 256 + merges 次数（再加 special tokens）。

**一次merge迭代的核心步骤：**
1. Step A：统计所有相邻 pair 的频率
    - 你要看每个 pre-token 的“当前 token 序列”（注意：随着 merge 进行，序列会变得更短、元素会变成“多字节 token”），然后对相邻对 (tok_i, tok_{i+1}) 做计数累加。 关键：要乘出现次数。
2. Step B：选出频率最高的 pair
    - 选 max count 的 pair。并且要遵守 PDF 的固定 `tie-break` 规则：如果频率并列，选 字典序更大（lexicographically greater） 的 pair，例如：
    ```python
    # 这四个字节对都具有最高频率，将合并('BA', 'A')，因为它在字典序上大于其他三个对。
    max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")]) # 输出('BA', 'A')
    ```
3. Step C：在所有 pre-token 中应用这次 merge
    - 对每个 pre-token 的 token 序列，把每个出现的 (A, B) 替换成一个新 token AB（概念上是把 bytes 拼接起来）。
4. Step D：把新 token 加入 vocab，并记录 merge
    - vocab 增加一个新 token（bytes串: AB）
    - merges表(即：规则表) 需要append (A, B)，并且顺序就是创建顺序（后续 encode 要按这个顺序应用）
    ```markdown
    # merges表示例
    [
    (b's', b't'),
    (b'e', b'st'),
    (b'l', b'ow'),
    ]
    ```
5. Step E：停止条件
    - 会一直循环直到：vocab size 达到 vocab_size 上限

```markdown
# 迭代示例
合并操作 我们首先查看每个连续的字节对，并统计它们出现在单词中的频率总和 {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}。其中 ('es') 和 ('st') 的频率相同，因此我们选择字典序较大的对 ('st')。然后我们会合并预标记，最终得到 {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}。

在第二轮中，我们发现 (e, st) 是最常见的对（出现次数为 9），并将其合并为 {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}。继续这一过程，最终得到的合并序列将是 ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']。


如果我们进行 6 次合并，得到 ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']，那么我们的词汇表元素将是 [<|endoftext|>, [...256 个字节字符], st, est, ow, low, west, ne]。

使用这个词汇表和merge规则集，单词"newest"将被分词为[ne, west]。
```

### 4. special token
**要求：**
- 在用 regex 预分词之前，要先把所有 special tokens 从语料中“剥离/切开”。
- 绝不能跨 special token 发生合并
- special token 必须加入词表（有固定 ID），一般在词表初始化时加入。

**原因：** 你绝对不能让 BPE 的 merge 跨越 <|endoftext|> 这种“文档边界”。举例：[Doc1]<|endoftext|>[Doc2] 必须拆成 Doc1 和 Doc2 分开预分词，保证不会跨文档合并。
**实现：** special token 必须加入词表（有固定 ID）

## 2.5 实验：BPE 分词器训练
1. 并行预分词
    - 预分词需要用 GPT-2 regex 在大文本上扫描，耗时大，因此成为主要瓶颈
    - 建议用 Python 自带 `multiprocessing` 并行，把语料切成多个 chunk 分发给多个进程做预分词统计，`(chunk 边界要落在某个 special token 的开头)`，最后把每个进程的统计结果合并起来。
    - 现成的 chunk 边界代码在`pretokenization_example.py的find_chunk_boundaries函数里`
2. 在预分词前去掉special token
    - 可以使用 `re.split`并以 `"|".join(special_tokens)` 作为分隔符实现，建议使用 `re.escape`（因为 `|` 等字符可能出现在 token 里）
3. 优化merge步骤
    - 可以通过缓存/索引 pair counts 并做增量更新来加速，而不是每次全量重算(了解即可)



## 小结

1. Unicode 解决字符编号。
2. UTF-8 解决编号到字节存储。
3. BPE 通过高频合并实现更短序列与更好泛化的平衡。
4. 工程实现重点在预分词边界、special token 隔离和 merge 统计效率。



# 示例：BPE 数据流

```text
{low: 5, lower: 2, widest: 3, newest: 6}
```

按你列的 1→6 步，把 **数据流（dataflow）** 和 **每一步的数据结构/键值类型** 讲清楚。并且我会特别强调：哪些地方是 `str`、哪些是 `bytes`、哪些是 `int`、以及为什么不能用“单个 byte 类型”。（PDF 里也强调 Python 没有单独 byte 类型、merge 不跨 pre-token 边界等点 ）

> 重要说明：你给的 `{low:5,...}` 在 PDF 例子里是“为了简单起见，假设 pretokenization 只按空格分割后得到的 pre-token 频率表”。真实作业用 GPT-2 regex，但数据流和类型是一样的。

---

## Step 1) 读原始文本（Unicode `str`）

**输入**：一整段文本文件内容（或流式读的 chunk），类型是：

* `corpus: str`

在 Python 里，`str` 是 Unicode 字符串（不是 bytes）。

---

## Step 2) 用 special token 做切分

假设 special token 有 `<|endoftext|>`。

你做的不是“删除它”，而是**把文本切成段**：

* `special_tokens: list[str]`
* `segments: list[str]`  （每段都是普通文本，不包含 special token）

切分原因：保证 **后续任何 merge 都不跨越 special token 边界**（文档边界）。

> 注意：special token 自己要进 vocab（有固定 id），但它**不参与**普通文本的 regex 预分词与 pair 统计（边界墙的作用）。

---

## Step 3) 对每段普通文本做 regex 预分词，得到 pre-token 字符串序列

对每个 `segment: str` 做 regex（GPT-2 pattern），输出是一串 **pre-token**，每个都是 `str`：

* `pretokens: list[str]`

例如（你的简化例子等价于空格分词的结果）：

* `"low"`、`"lower"`、`"widest"`、`"newest"` 都是 `str`

随后你会把相同 pre-token 聚合成计数表（这就是你给的那张表）：

* `pretok_counts_str: dict[str, int]`

也就是：

```python
{
  "low": 5,
  "lower": 2,
  "widest": 3,
  "newest": 6,
}
```

✅ 到这里为止：键是 `str`，值是 `int`。

---

## Step 4) 把每个 pre-token `str` encode 成 UTF-8 `bytes`，再转成“bytes 序列/tuple”

对每个 pre-token 字符串 `p: str`：

### 4.1 先编码成 UTF-8 bytes

* `b = p.encode("utf-8")`
* 类型：`b: bytes`

例如：

* `"low".encode("utf-8") == b"low"`
* `"newest".encode("utf-8") == b"newest"`

### 4.2 再把它变成“token 序列”

这里是关键：**BPE 一开始的 token 是“单字节 token”**。

但 Python 里：

* `list(b"low")` 得到的是 `[108, 111, 119]` ——元素是 `int`（这是很多人踩坑点）
* 但我们需要的是 **每个 token 是 `bytes`（长度为 1）**，而不是 `int`。PDF 特别提醒“Python 没有 byte 类型，单字节也用 bytes 表示”。

所以我们把 `"low"` 对应的初始 token 序列表示为：

* `tokens = (b"l", b"o", b"w")`
* 类型：`tuple[bytes, ...]`

同理：

* `"lower"` → `(b"l", b"o", b"w", b"e", b"r")`

✅ 到这里为止：你从 `str` 变成了 `tuple[bytes]`（每个元素是 bytes，不是 int）。

---

## Step 5) 统计每个 pre-token 的出现次数（计数表）

你原本有 `pretok_counts_str: dict[str, int]`，现在把 key 换成 bytes 序列形式：

* `word_counts: dict[tuple[bytes, ...], int]`

例如（这是你那张表“换成 bytes 序列 key”后的真实训练输入形态）：

```python
{
  (b"l", b"o", b"w"): 5,
  (b"l", b"o", b"w", b"e", b"r"): 2,
  (b"w", b"i", b"d", b"e", b"s", b"t"): 3,
  (b"n", b"e", b"w", b"e", b"s", b"t"): 6,
}
```

✅ 这一步的意义：之后你统计 pair 频率时，直接用这个 count 加权（不用扫全文）。

---

## Step 6) 后续 merge：只在每个 pre-token 内部统计 pair、合并 pair（禁止跨 pre-token）

### 6.1 pair 计数表长什么样？

从 `word_counts` 生成全局 pair 频率：

* `pair_counts: dict[tuple[bytes, bytes], int]`     

例如，对 `(b"l", b"o", b"w")`（count=5）贡献：

* `(b"l", b"o") += 5`
* `(b"o", b"w") += 5`

对 `(b"n", b"e", b"w", b"e", b"s", b"t")`（count=6）贡献：

* `(b"n", b"e") += 6`
* `(b"e", b"w") += 6`
* `(b"w", b"e") += 6`
* `(b"e", b"s") += 6`
* `(b"s", b"t") += 6`

**注意**：pair 只统计每个 tuple 内部相邻元素，不跨两个 pre-token 的边界。

### 6.2 一次 merge 会改变什么数据结构？

假设你选了最高频 pair `(A, B)`：

1. 把 merges 记录下来（顺序很重要）：

   * `merges: list[tuple[bytes, bytes]]`
   * append `(A, B)`（A、B 都是 `bytes`）

2. 把每个 word 的 token 序列更新成新序列（tuple 不可变，所以生成新 tuple）：

   * 原来 key：`(…, A, B, …)`
   * 新 key：`(…, AB, …)` 其中 `AB = A + B`（bytes 拼接）
   * count 不变（还是那个 `int` 次数）

3. vocab 也会增长（训练输出 vocab）：

   * `vocab: dict[int, bytes]`（id → token_bytes）
   * 新 token `AB` 会被分配一个新的 `int` id 并写入 vocab

> 这里你问的“键值对是什么类型”：
>
> * `word_counts` 的 key：`tuple[bytes,...]`
> * `word_counts` 的 value：`int`（频次）
> * `pair_counts` 的 key：`tuple[bytes,bytes]`
> * `pair_counts` 的 value：`int`
> * `merges` 的元素：`(bytes, bytes)`
> * `vocab`：`int -> bytes`

---

## 你特别关心的点：为什么不能用 int 来做 token (key)？

因为如果你把 token 当 int（0–255）：

* 合并后 token 可能是多字节序列（比如 `b"st"`），**已经不是单个 0–255 的数字能表示的**；
* 而用 `bytes` 表示 token（无论 1 字节还是多字节）就统一了：
  `b"s"`、`b"st"`、`b"newest"` 都是 `bytes`，类型一致，方便做 dict key、做拼接、做比较。

这也是 PDF 强调“单字节也用 bytes，没有 byte 标量类型”的原因。

---

## 小结：一张类型对照表

| 阶段         | 主要数据结构              | key 类型               | value 类型                   |
| ---------- | ------------------- | -------------------- | -------------------------- |
| 原文 | `corpus` | — | `str` |
| special 切分 | `segments` | — | `list[str]` |
| 预分词计数（概念） | `pretok_counts_str` | `str` | `int` |
| 转 UTF-8 | `b = p.encode()` | —  | `bytes` |
| token 序列化  | `tokens` | — | `tuple[bytes,...]` |
| 训练主输入表 | `word_counts` | `tuple[bytes,...]` | `int` |
| pair 统计 | `pair_counts` | `tuple[bytes,bytes]` | `int` |
| merges 历史  | `merges` | — | `list[tuple[bytes,bytes]]` |
| vocab 输出 | `vocab` | `int` | `bytes` |

---


