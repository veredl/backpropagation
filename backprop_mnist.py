
import numpy as np
import scipy.io
import scipy.optimize as op
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

 
# Import data
data = scipy.io.loadmat('dataset.mat')
x_digits = data["X"]
y_digits = data["y"]
# Correction:
y_digits[:500] = 0


# Normalization:
for i in range(len(x_digits)):                    
    x_digits[i,:] /= np.max(x_digits, axis = 1)[i]

# 3,000 / 5,000 digits for train, 1,000 for validation, 1,000 for final test
x_train, x_test, y_train, y_test = train_test_split(x_digits, y_digits, test_size = .4 , random_state = 42) 

x_validation, x_final_test, y_validation, y_final_test = train_test_split(x_test, y_test, test_size = .5 , random_state = 42) 



def initial_weights ():
    
    np.random.seed(42)
    
    b1 = np.random.normal(size=(1,25))
    b2 = np.random.normal(size=(1,10))
    w1 = np.random.normal(size=(400,25))
    w2 = np.random.normal(size=(25,10))
    
    return np.concatenate((b1, b2, w1, w2), axis=None)
    

def Gradient (theta, x_train, y_train, lamda = 0):
    
    b1 = theta[:25]
    b2 = theta[25:35]
    w1 = theta[35:10035].reshape(400,25)
    w2 = theta[10035:10285].reshape(25,10)
    
    dw_b1_avg = np.zeros((1,25))
    dw_b2_avg = np.zeros((1,10))    
    dw_w2_avg = np.zeros((25,10))
    dw_w1_avg = np.zeros((400,25))
    
    for i in range(len(x_train)):

        X = x_train[i,:]
        y = np.zeros(10)
        y[int(y_train[i])] = 1
        
        l1 = 1/(1+np.exp(-(np.dot(X,w1)+b1)))
        l2 = 1/(1+np.exp(-(np.dot(l1,w2)+b2)))
        
        l2_delta = (l2 - y) * (l2 * (1-l2))
        l1_delta = l2_delta.dot(w2.T) * (l1 * (1-l1))
        
        dw_w2 = l1.T.reshape(25,1).dot(l2_delta.reshape(1,10))
        dw_w1 = X.T.reshape(400,1).dot(l1_delta.reshape(1,25))

        dw_w2_avg += dw_w2/len(x_train) 
        dw_w1_avg += dw_w1/len(x_train) 
        dw_b1_avg += l1_delta/len(x_train)
        dw_b2_avg += l2_delta/len(x_train)
        
    dw_w2_avg += lamda*w2/len(x_train)
    dw_w1_avg += lamda*w1/len(x_train)
   
    w_derivatives = np.concatenate((dw_b1_avg, dw_b2_avg, dw_w1_avg, dw_w2_avg), axis=None)

    return w_derivatives


def CostFunc (theta, x_train, y_train, lamda = 0):
    
    b1 = theta[:25]
    b2 = theta[25:35]
    w1 = theta[35:10035].reshape(400,25)
    w2 = theta[10035:10285].reshape(25,10)
    
    loss = 0
    
    for i in range(len(x_train)):
        
        X = x_train[i,:]
        y = np.zeros(10)
        y[int(y_train[i])] = 1
        
        l1 = 1/(1+np.exp(-(np.dot(X,w1)+b1)))
        l2 = 1/(1+np.exp(-(np.dot(l1,w2)+b2)))
        
        loss += 0.5*np.sum((y - l2)**2)/len(x_train) 
        
    loss += 0.5*lamda/len(x_train)*(np.sum(w2**2)+np.sum(w1**2))

    return loss    


def check_backprop (x_train, y_train, minus = False):

    delta = 0.001
    
    np.random.seed(42)
    
    b1 = np.random.normal(size=(1,25))
    b2 = np.random.normal(size=(1,10))
    w1 = np.random.normal(size=(400,25))
    w2 = np.random.normal(size=(25,10))
        
    w_before = np.concatenate((b1, b2, w1, w2), axis=None)
    
    if minus == False:
        with_delta = np.tile(w_before, (len(w_before),1))+np.eye(len(w_before))*delta
    else:
        with_delta = np.tile(w_before, (len(w_before),1))-np.eye(len(w_before))*delta
    
    loss_with_delta = []
    
    for j in range (len(with_delta)):
        
        if j%1000 == 0:
            print('current w with delta is {}'.format(j))
        
        loss = 0
        
        b1_j = with_delta[j,:25] 
        b2_j = with_delta[j,25:35] 
        w1_j = with_delta[j,35:10035].reshape(400,25)
        w2_j = with_delta[j,10035:10285].reshape(25,10)
        
        for i in range(len(x_train)):
            
            X = x_train[i,:]
            y = np.zeros(10)
            y[int(y_train[i])] = 1
            
            l1 = 1/(1+np.exp(-(np.dot(X,w1_j)+b1_j)))
            l2 = 1/(1+np.exp(-(np.dot(l1,w2_j)+b2_j)))

            loss += 0.5*np.sum((y - l2)**2)/len(x_train)
            
        loss_with_delta.append(loss)

    return loss_with_delta


def run_backprop (iterations):

    lamda = 0
    loss_decline = []

    for i in iterations: 

        print('MaxIter is {}'.format(i))

        opt = op.fmin_cg(f = CostFunc, x0 = initial_theta, fprime = Gradient, args = (X, y, lamda), maxiter = i, full_output = True) 

        loss_decline.append(opt[1])

    return loss_decline


def validation_lamda (x_validation, y_validation, theta):
    
    b1 = theta[:25]
    b2 = theta[25:35]
    w1 = theta[35:10035].reshape(400,25)
    w2 = theta[10035:10285].reshape(25,10)
    
    loss = 0
    
    for i in range(len(x_validation)):
        
        X = x_validation[i,:]
        y = np.zeros(10)
        y[y_validation[i]] = 1
        
        l1 = 1/(1+np.exp(-(np.dot(X,w1)+b1)))
        l2 = 1/(1+np.exp(-(np.dot(l1,w2)+b2)))

        loss += 0.5*np.sum((y - l2)**2)/len(x_validation)

    return loss


def add_Reg (lamda_values):
    
    loss_per_lamda_Cvalidation = []
    loss_per_lamda_train = []    
    final_theta_per_lamda = []
    
    for lamda in lamda_values: 

        print('lamda is {}'.format(lamda))

        opt = op.fmin_cg(f = CostFunc, x0 = initial_theta, fprime = Gradient, args = (X, y, lamda), maxiter = 500, full_output = True)

        loss_per_lamda_train.append(opt[1])
        loss_per_lamda_Cvalidation.append(validation_lamda(x_validation, y_validation, opt[0]))
        final_theta_per_lamda.append(opt[0])
        
    return loss_per_lamda_Cvalidation, loss_per_lamda_train, final_theta_per_lamda


def success_rate_test (x_final_test, y_final_test, theta):
    
    b1 = theta[:25]
    b2 = theta[25:35]
    w1 = theta[35:10035].reshape(400,25)
    w2 = theta[10035:10285].reshape(25,10)
    
    success_rate = 0
    loss = 0

    for i in range(len(x_final_test)):
        
        X = x_final_test[i,:]
        y = np.zeros(10)
        y[y_final_test[i]] = 1
        
        l1 = 1/(1+np.exp(-(np.dot(X,w1)+b1)))
        l2 = 1/(1+np.exp(-(np.dot(l1,w2)+b2)))

        loss += 0.5*np.sum((y - l2)**2)/len(x_final_test)
        
        result = np.where(l2 == np.amax(l2))
        l2_b = np.zeros(10)
        l2_b[result] = 1 
        
        if not (y - l2_b).any():
            success_rate += 1/len(x_final_test)
        
    return loss, success_rate



if __name__ == "__main__":
    
    
    x_train_digit = x_train[2555,:].reshape(20,20).T
    y_train_digit = y_train[2555]
    plt.imshow(x_train_digit, cmap = 'gray', vmin = x_train_digit.min(), vmax = x_train_digit.max())
    sns.set(style='darkgrid')
    
    X = x_train
    y = y_train
    initial_theta = initial_weights()
       
    loss = CostFunc(initial_theta,X,y)
    w_derivatives = Gradient(initial_theta,X,y)
    
    loss_plus = check_backprop(X, y)
    loss_minus = check_backprop(X, y, True)
    dL_to_dw = (np.array(loss_plus) - np.array(loss_minus))/(2*0.001)    
        
    plt.figure()    
    plt.plot(w_derivatives)
    plt.title('Average w_derivatives calculated using BackPropagation, as function of w')
    plt.xlabel('w')
    plt.ylabel('w_derivatives')    
    plt.show()    
    
    plt.figure()
    plt.plot(dL_to_dw)
    plt.title('dL_to_dw calculated numerically, as function of w')
    plt.xlabel('w')
    plt.ylabel('dL_to_dw')
    plt.show()        
    
    plt.figure()
    plt.plot(dL_to_dw / w_derivatives)
    plt.title('dL_to_dw divided by w_derivatives')
    plt.xlabel('w')
    plt.ylabel('dL_to_dw / w_derivatives')    
    plt.show()  

    plt.figure()
    plt.plot(dL_to_dw - w_derivatives)
    plt.title('dL_to_dw minus w_derivatives')
    plt.xlabel('w')
    plt.ylabel('dL_to_dw - w_derivatives')    
    plt.show()      

    iterations = np.arange(10,220,10)
    loss_decline = run_backprop(iterations)
    
    plt.figure()
    plt.plot(iterations, loss_decline[:])
    plt.title('Cost Function, as a function of MaxIter')
    plt.xlabel('Maximum iterations')
    plt.ylabel('potential value')
    plt.show()    
    
    lamda_values = np.arange(0,1,0.1)
    loss_per_lamda_Cvalidation, loss_per_lamda_train, final_theta_per_lamda = add_Reg(lamda_values)
    
    plt.figure()
    plt.plot(lamda_values, np.array(loss_per_lamda_Cvalidation[:]))
    plt.title('Cost Function including regularization, as a function of lamda')
    plt.xlabel('lamda')
    plt.ylabel('potential value')
    plt.show()
    
    plt.figure()
    plt.plot(lamda_values, np.array(loss_per_lamda_train[:]))
    plt.title('Cost Function including regularization, as a function of lamda, on the train data')
    plt.xlabel('lamda')
    plt.ylabel('potential value')
    plt.show()

    result = np.where(loss_per_lamda_Cvalidation == np.amin(loss_per_lamda_Cvalidation))
    chosen_lamda = lamda_values [int(result[0])]
    f_w = final_theta_per_lamda[int(result[0])]
    final_loss, success_rate = success_rate_test (x_final_test, y_final_test, f_w)
    