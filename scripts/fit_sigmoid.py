#!/usr/bin/env python3
# Fit the sigmoid divisor D to a dataset's score scale: target = sigmoid(score/D),
# minimizing MSE against the WDL outcome. Records are 35B (32 pos + i16 score@32 + wdl@34).
# Coarse-then-fine search keeps it fast in pure Python (no numpy).
import struct, math, sys

path = sys.argv[1]
sample = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
with open(path, "rb") as f:
    data = f.read(35 * sample)
m = len(data) // 35
scores = [struct.unpack_from("<h", data, i * 35 + 32)[0] for i in range(m)]
outs = [1.0 if data[i * 35 + 34] == 0 else 0.0 if data[i * 35 + 34] == 1 else 0.5 for i in range(m)]

def mse(D):
    inv = 1.0 / D
    s = 0.0
    for sc, o in zip(scores, outs):
        s += (1.0 / (1.0 + math.exp(-sc * inv)) - o) ** 2
    return s

coarse = min(range(50, 1205, 25), key=mse)
fine = min(range(max(50, coarse - 25), coarse + 26, 5), key=mse)
bestD = min(range(max(50, fine - 5), fine + 6), key=mse)

abss = sorted(abs(s) for s in scores)
p99 = abss[int(0.99 * len(abss))] if abss else 0
print(f"sampled {m} positions from {path}")
print(f"  score |range| max={max(abs(s) for s in scores)} p99={p99}")
print(f"  wdl: win={outs.count(1.0)/len(outs):.3f} loss={outs.count(0.0)/len(outs):.3f} draw={outs.count(0.5)/len(outs):.3f}")
print(f"optimal sigmoid_divisor: {bestD}")
