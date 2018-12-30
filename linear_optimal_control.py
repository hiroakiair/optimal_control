##二次元離散システムの最適制御問題を解く

#numpyのimport
import numpy as np

#二次元離散システム(A,b)及びコスト関数(R,Q,N)の設定
A = np.array([[0.0, 1.0],[-1.0, 2.0]])
b = np.array([[0.0], [1.0]])
Q = np.array([[2.0, 1.0],[1.0, 3.0]])
R = np.array([[1.0]])
N = 100

#フィードバックゲインを各時間tでK=(K1[t], K2[t])で保持
t = np.arange(0, N, 1)
K = np.array([[0.0, 0.0]])
K1 = np.zeros(N)
K2 = np.zeros(N)

#リカッチ差分方程式を逐次的に解く
P = Q
for i in range(N):
    m = i+1
    S = np.dot(b.T, P)
    S = np.dot(S, b)
    T = np.dot(b.T, P)
    T = np.dot(T, A)
    K = -1 * np.dot(np.linalg.inv(S+R), T)
    K1[N - m] = K[0][0]
    K2[N - m] = K[0][1]
    P = np.dot(np.dot(A.T, P), A) + Q - np.dot(np.dot(np.dot(np.dot(A.T, P), b), np.linalg.inv(S+R)), T)

#最適コストの計算
J = 0
x = np.array([[2.0], [1.0]])
x1 = np.zeros(N)
x2 = np.zeros(N)
for i in range(N):
    x1[i] = x[0][0]
    x2[i] = x[1][0]
    K = [[K1[i], K2[i]]]
    u = np.dot(K, x)
    tmp = np.dot(np.dot(x.T, Q), x) + np.dot(u, u)
    J += tmp[0]
    x = np.dot(A, x) + np.dot(b, u)
    
#状態変数xの遷移
from matplotlib import pyplot
pyplot.plot(x1, x2)
pyplot.show()
#最適コスト
print(J)
print(K1[0])
print(K2[0])
