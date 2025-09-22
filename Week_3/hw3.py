# runge_joint_ultra_min_smooth.py
# Pure-Python, NO imports. Learn Runge f and f' jointly with a smooth activation.
# Loss = MSE(f) + MU * MSE(f')

# ----- target f and f' -----
def f(x):
    return 1.0 / (1.0 + 25.0 * x * x)

def fprime(x):
    d = 1.0 + 25.0 * x * x
    return -(50.0 * x) / (d * d)

# ----- data -----
def make_grid(n):
    xs = []
    if n == 1:
        xs = [0.0]
    else:
        for i in range(n):
            xs.append(-1.0 + 2.0 * i / (n - 1))
    ys  = [f(x) for x in xs]
    dys = [fprime(x) for x in xs]
    return xs, ys, dys

def make_midpoints(n):
    xs = []
    for i in range(n):
        xs.append(-1.0 + 2.0 * (i + 0.5) / n)
    ys  = [f(x) for x in xs]
    dys = [fprime(x) for x in xs]
    return xs, ys, dys

# ----- hyperparams -----
H      = 96      # hidden width (64~128皆可)
EPOCHS = 3200
LR0    = 0.01
MU     = 3.0     # 強化導數項權重
SEED   = 2025

x_tr, y_tr, dy_tr = make_grid(256)
x_va, y_va, dy_va = make_midpoints(256)

# ----- tiny RNG (no imports) -----
_seed = SEED
def rnd():
    global _seed
    _seed = (1103515245 * _seed + 12345) & 0x7fffffff
    return _seed / 2147483647.0

# ----- smooth activation: tanh-like rational (C^2) -----
# a(x)=27x + x^3, b(x)=27 + 9x^2, phi=a/b
def phi(x):
    return (27.0*x + x*x*x) / (27.0 + 9.0*x*x)

# phi'(x) = (729 - 162x^2 + 9x^4) / (27 + 9x^2)^2
def dphi(x):
    num = 729.0 - 162.0*x*x + 9.0*x*x*x*x
    den = (27.0 + 9.0*x*x); den = den*den
    return num / den

# phi''(x) = 3888 x (x^2 - 9) / (27 + 9x^2)^3
def ddphi(x):
    den = 27.0 + 9.0*x*x; den = den*den*den
    return 3888.0 * x * (x*x - 9.0) / den

# ----- params: 1 -> H -> 1 -----
W1 = [ (rnd()*4.0 - 2.0) for _ in range(H) ]             # U[-2,2]
b1 = [ (j/float(H) - 0.5) * 0.1 for j in range(H) ]      # small offsets
W2 = [ (rnd()*2.0 - 1.0) / (H**0.5) for _ in range(H) ]  # small
b2 = 0.0

def forward_single(x):
    # returns (z1[], a1[], yhat, yhat_dx)
    z1 = [W1[j]*x + b1[j] for j in range(H)]
    a1 = [phi(z1[j]) for j in range(H)]
    yhat = b2
    for j in range(H):
        yhat += W2[j] * a1[j]
    # dy/dx = sum_j W2_j * phi'(z1_j) * W1_j
    dydx = 0.0
    for j in range(H):
        dydx += W2[j] * dphi(z1[j]) * W1[j]
    return z1, a1, yhat, dydx

def mse(a, b):
    n = len(a); s = 0.0
    for i in range(n):
        d = a[i]-b[i]; s += d*d
    return s / n

# ----- training -----
lr = LR0
for ep in range(1, EPOCHS+1):
    dW2 = [0.0]*H; db2 = 0.0
    dW1 = [0.0]*H; db1 = [0.0]*H

    yh, dydyh = [], []
    N = len(x_tr); c_f = 2.0/N; c_d = 2.0*MU/N

    for i in range(N):
        x = x_tr[i]; y = y_tr[i]; yx = dy_tr[i]
        z1, a1, yhat, dydx = forward_single(x)
        yh.append(yhat); dydyh.append(dydx)

        dLdy  = c_f * (yhat - y)      # function loss grad
        dLddx = c_d * (dydx - yx)     # derivative loss grad

        # output layer
        for j in range(H):
            dW2[j] += dLdy * a1[j] + dLddx * (dphi(z1[j]) * W1[j])
        db2 += dLdy

        # hidden layer (include phi'' for derivative path)
        for j in range(H):
            # via function:
            g_fun = dLdy * W2[j] * dphi(z1[j])
            dW1[j] += g_fun * x
            db1[j] += g_fun
            # via derivative:
            g_der_W1 = dLddx * W2[j] * (ddphi(z1[j]) * x * W1[j] + dphi(z1[j]))
            g_der_b1 = dLddx * W2[j] * (ddphi(z1[j]) * W1[j])
            dW1[j] += g_der_W1
            db1[j] += g_der_b1

    # clip
    CLIP = 5.0
    for j in range(H):
        if dW1[j] >  CLIP: dW1[j] =  CLIP
        if dW1[j] < -CLIP: dW1[j] = -CLIP
        if dW2[j] >  CLIP: dW2[j] =  CLIP
        if dW2[j] < -CLIP: dW2[j] = -CLIP
    if db2 >  CLIP: db2 =  CLIP
    if db2 < -CLIP: db2 = -CLIP

    # SGD step
    for j in range(H):
        W2[j] -= lr * dW2[j]
    b2 -= lr * db2
    for j in range(H):
        W1[j] -= lr * dW1[j]
        b1[j] -= lr * db1[j]

    # logs (每 200 次印一次)
    if ep % 200 == 0:
        # quick val
        yh_v, dydyh_v = [], []
        for i in range(len(x_va)):
            z1,a1,yv,dyv = forward_single(x_va[i])
            yh_v.append(yv); dydyh_v.append(dyv)
        lf  = mse(yh, y_tr);       lf_v  = mse(yh_v, y_va)
        ld  = mse(dydyh, dy_tr);   ld_v  = mse(dydyh_v, dy_va)
        lt, lt_v = lf + MU*ld, lf_v + MU*ld_v
        print("epoch {}/{}  train (tot/f/df) = {:.6f}/{:.6f}/{:.6f}   val = {:.6f}/{:.6f}/{:.6f}"
              .format(ep, EPOCHS, lt, lf, ld, lt_v, lf_v, ld_v))

    # LR decay
    if ep in (1400, 2400):
        lr *= 0.5

# ----- final evaluation -----
xg = [ -1.0 + 2.0*i/999.0 for i in range(1000) ]
yg  = [ f(x) for x in xg ]
dgy = [ fprime(x) for x in xg ]
yhat_g, dyhat_g = [], []
for x in xg:
    z1,a1,yh,dyh = forward_single(x)
    yhat_g.append(yh); dyhat_g.append(dyh)

def max_abs_err(a,b):
    m=-1.0; idx=0
    for i in range(len(a)):
        e = a[i]-b[i];  e = -e if e<0.0 else e
        if e>m: m=e; idx=i
    return m, idx

def mse_list(a,b):
    n=len(a); s=0.0
    for i in range(n):
        d=a[i]-b[i]; s+=d*d
    return s/n

mse_f = mse_list(yhat_g, yg)
mse_d = mse_list(dyhat_g, dgy)
max_f, idx_f = max_abs_err(yhat_g, yg)
max_d, idx_d = max_abs_err(dyhat_g, dgy)

print("\n=== Final evaluation ===")
print("Function:   MSE = {:.8f}   MaxErr = {:.8f} at x={:.6f}".format(mse_f, max_f, xg[idx_f]))
print("Derivative: MSE = {:.8f}   MaxErr = {:.8f} at x={:.6f}".format(mse_d, max_d, xg[idx_d]))

print("\nPreview (x, f, yhat, f', yhat') first 8 rows:")
for i in range(8):
    print("{:.5f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}"
          .format(xg[i], yg[i], yhat_g[i], dgy[i], dyhat_g[i]))
