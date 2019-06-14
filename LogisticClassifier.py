import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):    
    f= open(filename,'r') 
    tmp_str=f.readline()
    tmp_arr=tmp_str[:-1].split(' ')
    N=int(tmp_arr[0]);n_row=int(tmp_arr[1]);n_col=int(tmp_arr[2])
    print("N=%d, row=%d, col=%d" %(N,n_row,n_col))
    data=np.zeros([N,n_row*n_col+1])
    for n in range(N):
        tmp_str=f.readline()
        tmp_arr=tmp_str[:-1].split(' ')       
        for i in range(n_row*n_col+1):
            data[n][i]=int(tmp_arr[i])
    f.close() 
    return N,n_row,n_col,data


def dataset():
    N,n_row,n_col,data=load_data('./AND.txt')

    X_train=data[:N,:-1];Y_train=data[:N,-1]
    

    return X_train, Y_train, N

def gradient(X, Xm, Y, Ym, N, alpha, lr):
    
    for n in range(N):
        value = (Ym - predictor(X, Xm, N, alpha)) * ((np.dot(Xm, X[n]) +1)**2)
        alpha[n] = alpha[n] + (lr*value)
    return alpha

def predictor(X, Xm, N, alpha):
    s = 0
    for n in range(N):
        s = s + ((np.dot(Xm, X[n]) +1)**2)*alpha[n]
    sigma = sigmoid(s)    
    return sigma 


def sigmoid(s):  
    large=30
    if s<-large: s=-large
    if s>large: s=large
    return (1 / (1 + np.exp(-s)))


def cost(X,Y,N, alpha):
    En=0;epsi=1.e-12
    for n in range(N):
        y_pred =predictor(X, X[n], N, alpha)
        if y_pred<epsi: y_pred=epsi
        if y_pred>1-epsi:y_pred=1-epsi
        En=En+Y[n]*np.log(y_pred)+(1-Y[n])*np.log(1-y_pred)
    En=-En/N
    return En


def stocastic(X, Y, N, max_it, lr):

    alpha = np.zeros(N) 

    error=[];error.append(cost(X,Y,N, alpha))

    epsi=0.5
    it=0
    while(error[-1]>epsi):
        
        idx = np.random.randint(0, N)
        
        Xm = X[idx, :]
        Ym = Y[idx]


        alpha = gradient(X, Xm, Y, Ym, N, alpha, lr)
        error.append(cost(X,Y,N, alpha))

        print('iteration %d, cost=%.2f\n' %(it,error[-1]),end='')

        #Second second codition to stop
        it=it+1   
        if(it>max_it): break
    return alpha, error


def training(X, Y, N, it, lr):

    alpha, error = stocastic(X,Y, N, it, lr)

    plot_cost(error)

    print('\nin-samples error=%.2f' % (error[-1]))

    return alpha

def validation(X, Y, N, alpha):

    predictions = []
    for n in range(N):
        y_pred = predictor(X, X[n], N, alpha)
        if y_pred > 0.5:
            y_pred = 1
        else:
            y_pred = 0
        
        prediction = []
        prediction.append(y_pred)
        prediction.append(Y[n])
        predictions.append(prediction)

    plot_prediction(predictions)

    print('out-samples error=%.2f\n' % (cost(X,Y,N, alpha)))

    return predictions


def confusion_matrix(predictions):
    TP = 0
    TN = 0 
    FN = 0
    FP = 0

    for prediction in predictions:
        #Counting the number of true positives and true negatives
        if prediction[0] == prediction[1]:
            if prediction[0] == 1 and prediction[1] == 1:
                TP += 1
            else:
                TN += 1
        else:
            #Counting the number of false positives and false negatives 
            if prediction[0] == 1 and prediction[1] == 0:
                FN += 1
            else:
                FP += 1

    
    matrix = np.matrix([[TP, FP], [FN, TN]])

    try:   
        recall = TP / (TP + FN)
    except Exception:
        recall = 0

    try:
        precision = TP / (TP + FP)
    except Exception:
        precision = 0
    
    try:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    except Exception:
        accuracy = 0

    return accuracy, precision, recall, matrix 


def plot_cost(error):
    plt.plot(range(len(error)), error, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.ylim([0,max(error)+1])
    plt.show()


def plot_prediction(predictions):
    y_test = []
    y_pred = []

    for prediction in predictions:
        y_test.append(prediction[1])
        y_pred.append(prediction[0])

    plt.subplot(2,1,1)
    plt.step(range(len(y_test)),y_test, where="mid", color='g')
    plt.xlabel('x_test')
    plt.title('REAL VALUES')
    plt.ylabel('y_test')

    plt.subplot(2,1,2)
    plt.step(range(len(y_pred)),y_pred, where="mid", color='r')
    plt.xlabel('x_test')
    plt.title('PREDICTION VALUES')
    plt.ylabel('y_prediction')
    plt.tight_layout()
    plt.show()

  
def main():

    X_train, Y_train, N = dataset()


    max_iterations = 450
    learning_rate = 0.02
    alpha = training(X_train, Y_train, N, max_iterations, learning_rate)

    predictions = validation(X_train, Y_train, N, alpha)

    accuracy, precision, recall, matrix = confusion_matrix(predictions)

    print("Accuracy: %.2f" % (accuracy))
    print("Precision: %.2f" % (precision))
    print("Recall: %.2f" % (recall))
    print("####Confusion Matrix####")
    print(matrix)
    print("########################")



main()