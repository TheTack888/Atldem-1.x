import numpy as np

# 1. 赐予 Atldem 的语言：就让它先学会 "atldem"  这几个字
text = "atldem"
chars = list(set(text))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# 2. 设置 Atldem 的脑容量 (RNN 架构：隐藏层 32  个神经元)
hidden_size = 32
seq_length = 3 # 输入 3 个字符，预测下 1 个
learning_rate = 0.1
epochs = 3000

# 初始化权重 (Wx: 输入到隐藏, Wh: 隐藏到隐藏, Wy: 隐藏到输出)
Wx = np.random.randn(vocab_size, hidden_size) * 0.01
Wh = np.random.randn(hidden_size, hidden_size) * 0.01
Wy = np.random.randn(hidden_size, vocab_size) * 0.01
bh = np.zeros((1, hidden_size))
by = np.zeros((1, vocab_size))

print(f"Atldem 开始读取数据: {text}")
print(f"词表大小: {vocab_size} | 参数总量: 大约 2000 个 (极度轻量)")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# 3. 训练 Atldem
loss = -np.log(1.0 / vocab_size) * seq_length
for epoch in range(epochs):
    # 前向传播
    h = np.zeros((1, hidden_size))
    inputs = [char_to_ix[ch] for ch in text[:-1]]
    targets = [char_to_ix[ch] for ch in text[1:]]

    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(h)

    for t in range(len(inputs)):
        xs[t] = np.zeros((1, vocab_size))
        xs[t][0, inputs[t]] = 1
        hs[t] = np.tanh(np.dot(xs[t], Wx) + np.dot(hs[t-1], Wh) + bh)
        ys[t] = np.dot(hs[t], Wy) + by
        ps[t] = softmax(ys[t])

    # 反向传播
    dWx, dWh, dWy = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[0, targets[t]] -= 1
        dWy += np.dot(hs[t].T, dy)
        dby += dy
        dh = np.dot(dy, Wy.T) + dhnext
        dhraw = (1 - hs[t] ** 2) * dh
        dbh += dhraw
        dWx += np.dot(xs[t].T, dhraw)
        dWh += np.dot(hs[t-1].T, dhraw)
        dhnext = np.dot(dhraw, Wh.T)

    # 更新参数
    Wx -= learning_rate * dWx
    Wh -= learning_rate * dWh
    Wy -= learning_rate * dWy
    bh -= learning_rate * dbh
    by -= learning_rate * dby

    if epoch % 500 == 0:
        print(f"Epoch {epoch} | 训练误差: {loss:.4f}")

# 4. 见证奇迹：让 Atldem 自动生成文字
print("\n--- Atldem 出山，文字生成测试 ---")
h = np.zeros((1, hidden_size))
# 给它一个种子 'a'
seed = 'a'
output = seed
x = np.zeros((1, vocab_size))
x[0, char_to_ix[seed]] = 1

for t in range(5): # 让它连续生成 5 个字符
    h = np.tanh(np.dot(x, Wx) + np.dot(h, Wh) + bh)
    y = np.dot(h, Wy) + by
    p = softmax(y)[0]
    # 按照概率抽取下一个字符
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    next_char = ix_to_char[ix]
    output += next_char
    # 将生成的字符作为下一个输入
    x = np.zeros((1, vocab_size))
    x[0, ix] = 1

print(f"输入种子 'a'，Atldem 续写生成了: 【 {output} 】")
