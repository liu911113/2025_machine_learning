# runge_nn_ultra_min.py
# Pure-Python, NO imports. 1-hidden-layer MLP (ReLU) to fit Runge f(x)=1/(1+25x^2) on [-1,1]

# ----- target function -----
def f(x):
    return 1.0 / (1.0 + 25.0 * x * x)

# uniform grid (train) + midpoints (val) to avoid duplicate x
def make_grid(n):
    xs = []
    if n == 1:
        xs = [0.0]
    else:
        for i in range(n):
            xs.append(-1.0 + 2.0 * i / (n - 1))
    ys = [f(x) for x in xs]
    return xs, ys

def make_midpoints(n):
    xs = []
    for i in range(n):
        xs.append(-1.0 + 2.0 * (i + 0.5) / n)
    ys = [f(x) for x in xs]
    return xs, ys

# ----- tiny MLP: 1 -> H -> 1 (ReLU) -----
H = 64          # hidden width
EPOCHS = 3000   # training steps
LR0 = 0.01      # initial learning rate
SEED = 12345    # for deterministic "random" (no imports)

# data
x_tr, y_tr = make_grid(256)
x_va, y_va = make_midpoints(256)

# simple LCG pseudo-random without imports
_seed = SEED
def rnd():
    global _seed
    _seed = (1103515245 * _seed + 12345) & 0x7fffffff
    return _seed / 2147483647.0  # in (0,1)

# init: spread W1, small bias offsets to break symmetry; small W2
W1 = [ (rnd()*4.0 - 2.0) for _ in range(H) ]            # ~U[-2,2]
b1 = [ (j / float(H) - 0.5) * 0.1 for j in range(H) ]   # small slope across units
W2 = [ (rnd()*2.0 - 1.0) / (H**0.5) for _ in range(H) ] # small
b2 = 0.0

# ReLU and derivative
def relu(x):  return x if x > 0.0 else 0.0
def d_relu(x): return 1.0 if x > 0.0 else 0.0

def forward_single(x):
    # returns (z1[], a1[], yhat)
    z1 = [W1[j]*x + b1[j] for j in range(H)]
    a1 = [relu(z1[j]) for j in range(H)]
    yhat = 0.0
    for j in range(H):
        yhat += W2[j] * a1[j]
    yhat += b2
    return z1, a1, yhat

def mse(yh, y):
    n = len(y)
    s = 0.0
    for i in range(n):
        d = yh[i] - y[i]
        s += d * d
    return s / n

train_loss, val_loss = [], []
lr = LR0

# ----- training (full-batch GD) -----
for ep in range(1, EPOCHS + 1):
    dW2 = [0.0] * H
    db2 = 0.0
    dW1 = [0.0] * H
    db1 = [0.0] * H

    yh_tr = []
    N = len(x_tr)
    invN2 = 2.0 / N  # for MSE gradient

    for i in range(N):
        x = x_tr[i]; y = y_tr[i]
        z1, a1, yhat = forward_single(x)
        yh_tr.append(yhat)

        dL_dy = invN2 * (yhat - y)  # dL/dyhat
        # output layer grads
        for j in range(H):
            dW2[j] += dL_dy * a1[j]
        db2 += dL_dy

        # hidden layer grads
        for j in range(H):
            da1 = dL_dy * W2[j]
            dz1 = da1 * d_relu(z1[j])
            dW1[j] += dz1 * x
            db1[j] += dz1

    # (optional) gradient clipping to avoid explosions with ReLU
    clip = 5.0
    for j in range(H):
        if dW1[j] >  clip: dW1[j] =  clip
        if dW1[j] < -clip: dW1[j] = -clip
        if dW2[j] >  clip: dW2[j] =  clip
        if dW2[j] < -clip: dW2[j] = -clip
    if db2 >  clip: db2 =  clip
    if db2 < -clip: db2 = -clip
    for j in range(H):
        if db1[j] >  clip: db1[j] =  clip
        if db1[j] < -clip: db1[j] = -clip

    # SGD step
    for j in range(H):
        W2[j] -= lr * dW2[j]
    b2 -= lr * db2
    for j in range(H):
        W1[j] -= lr * dW1[j]
        b1[j] -= lr * db1[j]

    # log
    tr = mse(yh_tr, y_tr)
    yh_va = [forward_single(x_va[i])[2] for i in range(len(x_va))]
    va = mse(yh_va, y_va)
    train_loss.append(tr); val_loss.append(va)

    if ep % 200 == 0:
        print("epoch {}/{}  train MSE={:.6f}  val MSE={:.6f}".format(ep, EPOCHS, tr, va))

    # simple LR decay
    if ep in (1200, 2200):
        lr *= 0.5

# ----- final eval -----
# dense grid (400 points)
xg = [ -1.0 + 2.0 * i / 399.0 for i in range(400) ]
yg = [ f(x) for x in xg ]
yg_hat = [ forward_single(x)[2] for x in xg ]

final_mse = mse(yg_hat, yg)
# max error & where
max_err = -1.0
max_i = 0
for i in range(len(xg)):
    e = yg_hat[i] - yg[i]
    if e < 0.0: e = -e
    if e > max_err:
        max_err = e
        max_i = i

print("\nFinal MSE={:.8f}  Max error={:.8f}".format(final_mse, max_err))
print("max err at x={:.6f}, true={:.6f}, pred={:.6f}".format(xg[max_i], yg[max_i], yg_hat[max_i]))

print("\nPreview (x, true, pred) first 10 rows:")
for i in range(10):
    print("{:.5f}, {:.6f}, {:.6f}".format(xg[i], yg[i], yg_hat[i]))

print("\nPreview (epoch, train_mse, val_mse) last 10 epochs:")
for i in range(max(0, len(train_loss)-10), len(train_loss)):
    print("{}, {:.6f}, {:.6f}".format(i+1, train_loss[i], val_loss[i]))
