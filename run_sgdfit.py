import cffi
import numpy as np
import threading
from tic import tic

ffi = cffi.FFI()
ffi.cdef("""float fit(unsigned int d, unsigned int N, float *x, float *y,
		     unsigned int nofs, size_t *offsets, size_t *counts,
		     unsigned int T, float alpha);""")
C = ffi.dlopen('maxV.dylib')

# X_train = np.load('X_train.npy')
# #X_norm = np.reshape((X_train[:, 0]/X_train[:, 1]), (X_train.shape[0], 1))
# X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
# X_train = np.ascontiguousarray(X_train.T, dtype=np.float32)

# y_train = np.load('y_train.npy')
# y_train = np.ascontiguousarray(y_train.T, dtype=np.float32)
# ofs_train = np.load('ofs_train.npy')
# r_train = np.load('r_train.npy')
# k_train = np.load('k_train.npy')

# w = np.mat([1, -0.5])
# y_train = np.ascontiguousarray(2*(np.dot(w, X_train) > 0)-1, dtype=np.float32)

# X_train = np.array([1, 2, 3, 4, 5, 6])
# X_train.shape = (2, 3)
# y_train = np.array([1, -1, 1], dtype=np.float32)


#feat_idx = [4, 10, 11, 14, 23, 24, 25, 33, 34, 35, 36, 37]
feat_idx = [4, 10, 11, 14, 15, 17, 22, 23, 25, 34, 35, 36, 37]

def poly2(X):
    n = X.shape[1]
#    m = int((n**2/2) if n % 2 == 0 else n**2/2 - n - 1)
    m = 0
    for i in range(0, n):
        for j in range(i, n):
            m += 1
    Y = np.empty((X.shape[0], m))
    k = 0
    for i in range(0, n):
        for j in range(i, n):
            print((k, i, j))
            Y[:, k] = X[:, i] * X[:, j]
            k += 1
    return Y

N = 1_000_000
with tic:
    X_train = np.load('features.npy')
#    X_train = np.hstack([X_train[:, 4:12], X_train[:, 14:]])
    X_train = X_train[0:N, feat_idx]
    # Xmean = np.mean(X_train, axis=0)
    # X_train = X_train - Xmean
    # Xstd = np.std(X_train, axis=0)
    # Xstd[Xstd == 0] = 1
    # X_train = X_train / Xstd
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train, poly2(X_train)])
#    X_train = poly2(X_train)    
    #X_norm = np.reshape((X_train[:, 0]/X_train[:, 1]), (X_train.shape[0], 1))
    #X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), 1)
    X_train = np.ascontiguousarray(X_train.T, dtype=np.float32)

    y_train = np.load('labels.npy')
    y_train = y_train[0:N, :]
    y_train = np.ascontiguousarray(y_train.T, dtype=np.float32)
    y_train[y_train == 0] = -1
    ofs_train = np.load('offsets.npy')
    ofs_train = np.ascontiguousarray(ofs_train, dtype=np.int64)
    r_train = np.load('counts.npy')
    r_train = np.ascontiguousarray(r_train, dtype=np.int64)
    ofs_idx = (ofs_train + r_train) < N
    ofs_train = ofs_train[ofs_idx]
    r_train = r_train[ofs_idx]


#print(X_train)
#exit()
print(np.max(ofs_train + r_train))
N = X_train.shape[1]
d = X_train.shape[0]
nofs = len(ofs_train)
w = np.ones(d) * 1e-6
w[5] = 1
w[13] = -1
y_train = np.sign(np.dot(w, X_train))
y_train[y_train == 0] = -1


X_train_ptr = ffi.cast("float *", ffi.from_buffer(X_train))
y_train_ptr = ffi.cast("float *", ffi.from_buffer(y_train))
print(y_train)
ofs_train_ptr = ffi.cast("size_t *", ffi.from_buffer(ofs_train))
r_train_ptr = ffi.cast("size_t *", ffi.from_buffer(r_train))

T = 1000000
alpha = 1
#nofs = 1
#r_train[0] = 3
print("{} training instances".format(nofs))
with tic:
    C.fit(d, N, X_train_ptr, y_train_ptr, nofs, ofs_train_ptr, r_train_ptr, T, alpha)
