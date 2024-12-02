# RSA-OAEP

## MGF

1. **初始化消息摘要**:
   - `MessageDigest md = MessageDigest.getInstance("SHA-256");`: 使用 SHA-256 算法创建一个消息摘要实例。

2. **准备变量**:
   - `byte[] T = new byte[maskLen];`: 创建一个长度为 `maskLen` 的数组 `T`，用于存储最终的掩码。
   - `int counter = 0;`: 初始化计数器。
   - `byte[] C = new byte[4];`: 创建一个 4 字节的数组 `C`，用于存储计数器的值。
   - `int hLen = md.getDigestLength();`: 获取 SHA-256 摘要的长度。

3. **生成掩码**:
   - 循环直到生成足够长度的掩码：
     - `ByteBuffer.wrap(C).putInt(counter);`: 将计数器的值转换为字节并存储在 `C` 中。
     - `md.update(X); md.update(C);`: 更新消息摘要对象，先加入输入种子 `X`，然后加入计数器 `C`。
     - `System.arraycopy(md.digest(), 0, T, counter * hLen, hLen);`: 将消息摘要的结果复制到 `T` 数组中，形成掩码的一部分。
     - `counter++;`: 增加计数器。

4. **处理剩余部分**:
   - 如果 `maskLen` 不能被 `hLen` 整除，处理最后一部分：
     - 再次更新消息摘要对象，并将最后一部分的摘要结果复制到 `T`。

5. **返回掩码**:
   - 返回生成的掩码 `T`。

## 填充

1. **初始化消息摘要**:
   - `MessageDigest md = MessageDigest.getInstance("SHA-256");`: 使用 SHA-256 算法创建消息摘要实例。

2. **计算 L 的哈希值**:
   - `byte[] lHash = md.digest(L.getBytes());`: 计算标签 `L` 的哈希值。

3. **检查消息长度**:
   - 如果消息长度超过允许的最大长度，抛出异常。

4. **准备数据块 DB**:
   - `byte[] PS = new byte[k - mLen - 2 * hLen - 2];`: 创建一个填充字符串 `PS`。
   - `byte[] DB = ...`: 创建数据块 `DB`，包含 `lHash`、`PS`、一个字节的 `0x01` 和原始消息。

5. **生成种子和掩码**:
   - `byte[] seed = new byte[hLen];`: 创建随机种子。
   - `new SecureRandom().nextBytes(seed);`: 使用安全随机数生成器填充种子。
   - `byte[] dbMask = MGF(seed, k - hLen - 1);`: 使用掩码生成函数（MGF）生成数据块掩码。

6. **掩码数据块 DB**:
   - `byte[] maskedDB = new byte[dbMask.length];`: 创建掩码后的数据块。
   - 使用异或操作将 `dbMask` 应用于 `DB`。

7. **掩码种子**:
   - `byte[] seedMask = MGF(maskedDB, hLen);`: 使用 `maskedDB` 生成种子掩码。
   - `byte[] maskedSeed = new byte[seed.length];`: 创建掩码后的种子。
   - 使用异或操作将 `seedMask` 应用于 `seed`。

8. **组合编码消息**:
   - 将一个字节的 `0x00`、`maskedSeed` 和 `maskedDB` 组合成最终的编码消息。