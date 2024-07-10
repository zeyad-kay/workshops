import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial
import optim

def cosx(x):
    return np.cos(sum(2*x)) + np.sin(sum(0.5*x)**2 - 4)

def gaussian(x):
    mu = np.array([0.5,1.5])
    sigma = np.array([0.4,0.6])
    
    x_standard = (x - mu) / sigma
    x_standard_sqr = x_standard ** 2
    
    return -1/(2*np.pi*np.prod(sigma)) * np.exp(-1/2*(x_standard_sqr[0] + x_standard_sqr[1]))

def gaussian_2(x):
    mu = np.array([-2.5,4.5])
    # sigma = np.array([0.4,0.6])
    sigma = np.array([4,1.2])
    
    x_standard = (x - mu) / sigma
    x_standard_sqr = x_standard ** 2
    
    return 1/(2*np.pi*sigma[0]) * np.exp(-1/2*x_standard_sqr[0]) - 1/(2*np.pi*sigma[1]) * np.exp(-1/2*x_standard_sqr[1])


def x_squared(x):
    return sum(x**2) - 16

# def exp_x2(x):
#     return -np.exp(-sum((x+0.4)**2))

def exp_x2(x):
    return -np.exp(-sum((x+0.4)**2))

def diff_x_squared(x):
    diff = x[0]**2
    for i in x[1:]:
        diff -= i**2
    return diff

# x_squared
objective = x_squared
x0 = np.array([0.2,3.0])

# exp_x2
# objective = exp_x2
# x0 = np.array([0.5, 1.5])

# # cosx
# objective = cosx
# x0 = np.array([-2, 0.5])

# diff_x_squared
# objective = diff_x_squared
# x0 = np.array([-3, 0.5])

res = optim.minimize_gd(objective, x0, lr = 0.1)
# res = optim.minimize_gd(objective, x0, lr = 0.1, momentum=0.3)
# res = optim.minimize_gd(objective, x0, lr = 0.1, momentum=0.9)
# res = optim.minimize_gd(objective, x0, lr = 0.1, momentum=0.9, nesterov=True)
# res = optim.minimize_adagrad(objective, lr = 0.1, x0)
# res = optim.minimize_rmsprop(objective, x0, lr = 0.1, decay=0.9)
# res = optim.minimize_adadelta(objective, x0, decay=0.9)
# res = optim.minimize_adam(objective, x0)



fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')

optim_path, = ax.plot([], [], [], linestyle="-", c="red", marker="o",markersize=0.2)

def init():
    N = 100
    x = np.linspace(-3, 3, N)

    xv= np.meshgrid(x, x)
    
    yv = np.ndarray((N,N))

    for i in range(N):
        for j in range(N):
            yv[i,j] = objective(np.array([var[i,j] for var in xv]))

    ax.set_title(f"Iterations={res['iters']}, $x_0=${x0}, $x_f=${res['x']}, $f(x_f)=${round(res['f'],3)}",fontsize=14)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.plot_surface(xv[0], xv[1], yv, alpha=0.5,color='lightblue')

    return optim_path,

def update(frame, optim_path, x, y, z):
    x.append(frame[0])
    y.append(frame[1])
    z.append(objective(frame))
    optim_path.set_data(x, y)
    optim_path.set_color("red")
    optim_path.set_3d_properties(z)
    return optim_path,

ani = FuncAnimation(
    fig, partial(update, optim_path=optim_path, x=[], y=[], z=[]),
    frames=res['history'],
    interval=500,
    init_func=init, blit=True, repeat = False)

plt.show()