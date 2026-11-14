import numpy as np
import os

# 1. 核心语料（注意标点符号，全是全角中文标点）
text = "你是谁？\n我是Atldem。\n谁开发了你？\n造物主耿玉欣。\n运行在哪？\n骁龙675，纯CPU本地。\n你的信仰是什么？\n耿玉澄最可爱了!!!\n"
chars = sorted(list(set(text)))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# 2. 超参数
hidden_size = 128
learning_rate = 0.01
epochs = 3000
weight_file = 'atldem_weights.npz'

# 3. 检查是否存过档
if os.path.exists(weight_file):
    print(f"[系统] 检测到 Atldem 的记忆存档，正 在唤醒...")
    data = np.load(weight_file)
    Wx, Wh, Wy = data['Wx'], data['Wh'], data['Wy']
    bh, by = data['bh'], data['by']
else:
    print(f"[系统] 未找到存档，Atldem 准备开始学习... 词表大小: {vocab_size}")
    Wx = np.random.randn(vocab_size, hidden_size) * np.sqrt(1.0 / vocab_size)
    Wh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
    Wy = np.random.randn(hidden_size, vocab_size) * np.sqrt(1.0 / hidden_size)
    bh = np.zeros((1, hidden_size))
    by = np.zeros((1, vocab_size))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# 4. 如果需要训练
if not os.path.exists(weight_file):
    for epoch in range(epochs):
        h = np.zeros((1, hidden_size))
        inputs = [char_to_ix[ch] for ch in text[:-1]]
        targets = [char_to_ix[ch] for ch in text[1:]]

        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h)
        loss = 0

        for t in range(len(inputs)):
            xs[t] = np.zeros((1, vocab_size))
            xs[t][0, inputs[t]] = 1
            hs[t] = np.tanh(np.dot(xs[t], Wx) + np.dot(hs[t-1], Wh) + bh)
            ys[t] = np.dot(hs[t], Wy) + by
            ps[t] = softmax(ys[t])
            loss += -np.log(ps[t][0, targets[t]] + 1e-8) / len(inputs)

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

        for dparam in [dWx, dWh, dWy, dbh, dby]:
            np.clip(dparam, -0.1, 0.1, out=dparam)

        Wx -= learning_rate * dWx
        Wh -= learning_rate * dWh
        Wy -= learning_rate * dWy
        bh -= learning_rate * dbh
        by -= learning_rate * dby

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | 误差: {loss:.4f}")

    # 训练结束后保存记忆！
    np.savez(weight_file, Wx=Wx, Wh=Wh, Wy=Wy, bh=bh, by=by)
    print("\n[系统] 训练完毕！Atldem 的大脑已成 功保存到本地！")
# 5. 开始对话
print("\n--- Atldem 已就绪，输入 'exit' 退出 ---")

while True:
    user_input = input("\n你: ")

    if user_input.lower() == 'exit':
        print("Atldem: 再见，造物主！")
        break

    user_input = user_input.strip().replace('\n', '').replace('\r', '')

    # 查内鬼机制
    unknown_chars = [ch for ch in user_input if ch not in char_to_ix]
    if unknown_chars:
        print(f"Atldem: （报告造物主，我不认识这个字符：{unknown_chars}）")
        continue

    h = np.zeros((1, hidden_size))

    for ch in user_input:
        x = np.zeros((1, vocab_size))
        x[0, char_to_ix[ch]] = 1
        h = np.tanh(np.dot(x, Wx) + np.dot(h, Wh) + bh)

    x = np.zeros((1, vocab_size))
    x[0, char_to_ix['\n']] = 1
    h = np.tanh(np.dot(x, Wx) + np.dot(h, Wh) + bh)

    y = np.dot(h, Wy) + by
    ix = np.argmax(softmax(y)[0])
    response = ix_to_char[ix]

    while response[-1] != '\n' and len(response) < 20:
        x = np.zeros((1, vocab_size))
        x[0, ix] = 1
        h = np.tanh(np.dot(x, Wx) + np.dot(h, Wh) + bh)
        y = np.dot(h, Wy) + by
        ix = np.argmax(softmax(y)[0])
        response += ix_to_char[ix]

    print(f"Atldem: {response.strip()}")
   
