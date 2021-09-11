import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import dft


def part_1_1():
    lamda = [0.01, 0.05, 0.1, 0.2]
    results = np.zeros([8], dtype=float)
    x_i = np.zeros([1, 128], dtype=float)
    for l in range(len(lamda)):
        for i in range(128):
            if abs(y[0, i]) > lamda[l]:
                x_i[0, i] = y[0, i] * (1 - lamda[l] / abs(y[0, i]))
            else:
                x_i[0, i] = 0
        results[l] = np.linalg.norm(y[0] - x_i[0])
        results[l + 4] = np.linalg.norm(x_i[0] - vector_x[0])
    plt.plot([0.01, 0.05, 0.1, 0.2], results[:4], marker='o')
    plt.plot([0.01, 0.05, 0.1, 0.2], results[4:], marker='o')
    plt.legend(['||x^ - y||', '||x^ - x||'])
    plt.xlabel('Lambda values')
    plt.ylabel("L2-norm")
    plt.title('Error Plot for lambdas')
    plt.show()

    fig, axs = plt.subplots(2)
    fig.suptitle('x vectors')
    axs[0].plot(vector_x[0], color='b', label='Actual x')
    axs[0].legend(loc="upper right")
    axs[1].plot(x_i[0], color='g', label='Obtained x')
    axs[1].legend(loc="upper right")
    plt.show()


def part_1_2():
    fftc_X = np.fft.fftshift(np.fft.fft(np.fft.fftshift(vector_x[0])))
    X_u = np.zeros([1, 128], dtype=complex)
    X_u[0, 0:128:4] = fftc_X[0:128:4]
    x_u = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(X_u))) * 4

    Fc = dft(128)
    Fc = np.roll(Fc, 64)
    M = np.zeros((128, 128))
    for i in range(0, 128, 4):
        M[i][i] = 1
    Fu = M @ Fc
    x_u_mnls = np.dot(np.linalg.pinv(Fu), fftc_X.T)
    print(np.linalg.norm(x_u[0] / 4 - x_u_mnls))

    X_r = np.zeros([1, 128], dtype=complex)
    prm = np.random.permutation(128)
    X_r[0, prm[:32]] = fftc_X[prm[:32]]
    x_r = np.fft.fftshift(np.fft.ifft(np.fft.fftshift(X_r))) * 4

    for i in range(len(x_r[0])):
        if x_r[0, i] < 0.2:
            x_r[0, i] = 0

    plt.plot(vector_x[0])
    plt.legend(['actual_sparse_x'])
    plt.xlabel('X indexes')
    plt.ylabel("|x|")
    plt.title('Different X values')
    plt.show()
    plt.plot(abs(x_u[0]))
    plt.legend(['x_u'])
    plt.xlabel('X indexes')
    plt.ylabel("|x|")
    plt.title('Different X values')
    plt.show()
    plt.plot(abs(x_r[0]))
    plt.legend(['x_r'])
    plt.xlabel('X indexes')
    plt.ylabel("|x|")
    plt.title('Different X values')
    plt.show()

    fig, axs = plt.subplots(2)
    fig.suptitle('Plots of x_u/4 and x_u_mnls')
    axs[0].plot(abs(x_u[0] / 4), color='b', label='x_u / 4')
    axs[0].legend(loc="upper right")
    axs[1].plot(abs(x_u_mnls), color='g', label='x_u_mnls')
    axs[1].legend(loc="upper right")
    plt.show()


def ADMM(A, b, lamda, rho, iter):
    z = 0.001 * np.random.randn(128)
    u = 0.001 * np.random.randn(128)
    error = []
    for k in range(iter):
        x_hat = np.linalg.inv(A.conj().T @ A + rho * np.identity(128)) @ (np.dot(A.conj().T, b) + rho * (z - u))
        magnitude = np.maximum(abs(x_hat + u) - lamda / rho, 0)
        phase = np.angle(x_hat + u)
        z = np.exp(1j * phase) * magnitude
        u = u + x_hat - z
        error.append(np.linalg.norm(vector_x - abs(x_hat)))
    return x_hat, error


def part_1_3():
    Fc = dft(128)
    Fc = np.roll(Fc, 64)
    M = np.zeros((128, 128))
    prm = np.random.permutation(128)
    for i in range(32):
        M[prm[i]][prm[i]] = 1
    Fu = M @ Fc
    x_hat1, error1 = ADMM(Fu, Y, 0.01, 10, 40)
    x_hat2, error2 = ADMM(Fu, Y, 0.05, 10, 40)
    x_hat3, error3 = ADMM(Fu, Y, 0.1, 10, 40)

    plt.plot(error1)
    plt.plot(error2)
    plt.plot(error3)
    plt.legend(['lambda: 0.01', 'lambda: 0.05', 'lambda: 0.1'])
    plt.xlabel('Iteration')
    plt.ylabel("L2 norm of (x^ - x)")
    plt.title('Error Plot for Lambdas')
    plt.show()

    fig, axs = plt.subplots(4)
    fig.suptitle('Plots of Different x values')
    axs[0].plot(vector_x[0], color='b', label='Created x')
    axs[0].legend(loc="upper right")
    axs[1].plot(abs(x_hat1), color='g', label='Predicted x for lambda: 0.01')
    axs[1].legend(loc="upper right")
    axs[2].plot(abs(x_hat2), color='r', label='Predicted x for lambda: 0.05')
    axs[2].legend(loc="upper right")
    axs[3].plot(abs(x_hat3), color='m', label='Predicted x for lambda: 0.1')
    axs[3].legend(loc="upper right")
    plt.show()


vector_x = np.zeros([1, 128], dtype=float)
vector_x[0, 0] = 1 / 5
vector_x[0, 1] = 2 / 5
vector_x[0, 2] = 3 / 5
vector_x[0, 3] = 4 / 5
vector_x[0, 4] = 1
np.random.shuffle(vector_x[0, :])
sigma = 0.05
n = sigma * np.random.randn(128)
y = vector_x + n
Y = np.fft.fftshift(np.fft.fft(np.fft.fftshift(y[0])))

part_1_1()
part_1_2()
part_1_3()