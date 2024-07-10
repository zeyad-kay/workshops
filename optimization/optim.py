import numpy as np

def gradient_nd(f, x0, e=0.001):
    
    return np.array([gradient_1d(f, np.array([x]), e) for x in x0])

def gradient_1d(f, x0, e=0.001):

    return (f(x0 + e) - f(x0 - e)) / (2*e)

def minimize_gd(f, x0, max_iters=1000, e=0.001, lr = 0.1, momentum=0.0, nesterov=False):

    history = [x0]

    grad = float('inf')
    prev_grad = None
    
    for i in range(1,max_iters+1):
        grad = gradient_nd(f, x0, e)
        
        if momentum != 0:
            if prev_grad is not None:
                grad_momentum = momentum * prev_grad + grad
            else:
                grad_momentum = grad

            prev_grad = grad_momentum

            if nesterov:
                grad = grad + momentum * grad_momentum
            else:
                grad = grad_momentum

        x0 = x0 - lr * grad
        
        history.append(x0)

        if (np.abs(grad) < e).all():
            break
    
    return {
        "history": np.array(history).round(3),
        "x": np.round(x0,3),
        "f": f(np.round(x0,3)),
        "iters": i
    }

def minimize_adagrad(f, x0, max_iters=1000, e=0.001, lr = 0.1):

    history = [x0]

    grad = float('inf')
    
    sum_squared_grad = np.zeros(x0.shape)

    for i in range(1,max_iters+1):

        grad = gradient_nd(f, x0, e)
        
        sum_squared_grad = sum_squared_grad + grad**2

        # 1e-8 prevents division by 0
        x0 = x0 - lr / (np.sqrt(sum_squared_grad) + 1e-8) * grad

        history.append(x0)

        if (np.abs(grad) < e).all():
            break
    
    return {
        "history": np.array(history).round(3),
        "x": np.round(x0,3),
        "f": f(np.round(x0,3)),
        "iters": i
    }   

def minimize_rmsprop(f, x0, max_iters=1000, e=0.001, lr = 0.1, decay=0.9):

    history = [x0]

    grad = float('inf')
    
    sum_squared_grad = np.zeros(x0.shape)

    for i in range(1,max_iters+1):

        grad = gradient_nd(f, x0, e)
        
        sum_squared_grad = decay * sum_squared_grad + (1 - decay) * grad**2

        # 1e-8 prevents division by 0
        x0 = x0 - lr / (np.sqrt(sum_squared_grad) + 1e-8) * grad

        history.append(x0)

        if (np.abs(grad) < e).all():
            break
    
    return {
        "history": np.array(history).round(3),
        "x": np.round(x0,3),
        "f": f(np.round(x0,3)),
        "iters": i
    }

def minimize_adadelta(f, x0, lr=1, max_iters=1000, e=0.001, decay=0.9):

    history = [x0]

    grad = float('inf')
    
    sum_squared_grad = np.zeros(x0.shape)
    sum_squared_delta_x = np.zeros(x0.shape)

    for i in range(1,max_iters+1):
        grad = gradient_nd(f, x0, e)

        sum_squared_grad = decay * sum_squared_grad + (1 - decay) * grad**2

        # 1e-8 prevents division by 0
        delta_x = (np.sqrt(sum_squared_delta_x) + 1e-3) / (np.sqrt(sum_squared_grad) + 1e-8) * grad

        sum_squared_delta_x = decay * sum_squared_delta_x + (1 - decay) * delta_x **2

        x0 = x0 - lr*delta_x

        history.append(x0)

        if (np.abs(grad) < e).all():
            break
    
    return {
        "history": np.array(history).round(3),
        "x": np.round(x0,3),
        "f": f(np.round(x0,3)),
        "iters": i
    }

def minimize_adam(f, x0, lr=0.1, betas=(0.9,0.999), max_iters=1000, e=0.001):

    history = [x0]

    grad = float('inf')
    
    sum_grad = np.zeros(x0.shape)
    sum_squared_grad = np.zeros(x0.shape)

    for i in range(1,max_iters+1):
        grad = gradient_nd(f, x0, e)
        
        sum_grad = betas[0] * sum_grad + (1 - betas[0]) * grad
        sum_squared_grad = betas[1] * sum_squared_grad + (1 - betas[1]) * grad**2

        corrected_sum_grad = sum_grad / (1-betas[0]**i)
        corrected_sum_squared_grad = sum_squared_grad / (1-betas[1]**i)

        # 1e-8 prevents division by 0
        delta_x = lr * corrected_sum_grad / (np.sqrt(corrected_sum_squared_grad) + 1e-8)
        x0 = x0 - delta_x

        history.append(x0)

        if (np.abs(grad) < e).all():
            break
    
    return {
        "history": np.array(history).round(3),
        "x": np.round(x0,3),
        "f": f(np.round(x0,3)),
        "iters": i
    }