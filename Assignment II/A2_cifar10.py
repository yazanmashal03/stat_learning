import numpy as np 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

dict1 = unpickle("./data/cifar-10-batches-py/data_batch_1")
dict2 = unpickle("./data/cifar-10-batches-py/data_batch_2")
dict3 = unpickle("./data/cifar-10-batches-py/data_batch_3")
dict4 = unpickle("./data/cifar-10-batches-py/data_batch_4")
dict5 = unpickle("./data/cifar-10-batches-py/data_batch_5")
test = unpickle("./data/cifar-10-batches-py/test_batch")
meta_data = unpickle("./data/cifar-10-batches-py/batches.meta")
label_names = meta_data["label_names"]


X_train = np.concatenate((dict1["data"],dict2["data"],dict3["data"],dict4["data"],dict5["data"]))
y_train = np.concatenate((dict1["labels"],dict2["labels"],dict3["labels"],dict4["labels"],dict5["labels"]))
X_test = test["data"]
y_test = test["labels"]

def data_to_image(x):
    return(x.reshape(3,32,32).transpose(1,2,0))

def plot_image(image, title=""):
    fig = plt.imshow(data_to_image(image))
    plt.title(title)
    fig.axes.set_axis_off()
    plt.show()

# as a verification that everything is working correctly, plot an image
image_nr = 3141
plot_image(X_train[image_nr,:],label_names[y_train[image_nr]])


## B - Fitting a logistic Regression model
model = linear_model.LogisticRegression(penalty="l2",C=1e-9).fit(X_train, y_train)
result = model.predict(X_test)
print(sum(result == y_test))
print(len(y_test))
print(X_train.shape)

print(label_names[0])

## C - Crossvalidation
# note that in the choosing of folds, there is some randomness
X_train_temp = X_train[:1000,:]
y_train_temp = y_train[:1000]
K = 4
C_list = [1e-6,1e-7,1e-8,1e-9,1e-10]
model = linear_model.LogisticRegressionCV(cv = K, Cs = C_list).fit(X_train_temp,y_train_temp)
fold_accuracies = model.scores_[0]
mean_accuracies = np.mean(fold_accuracies, axis=0)
print(fold_accuracies)
print(mean_accuracies)

plt.figure()
plt.loglog(C_list,mean_accuracies)
plt.show()

## D - Crossvalidation with logarithmic scoring rule
K = 4
C_list = [1e-6,1e-7,1e-8,1e-9,1e-10]
acc_arr = np.array([])
for C in C_list:
    model = linear_model.LogisticRegression(C=C)
    acc = model_selection.cross_val_score(model, X=X_train_temp, y=y_train_temp, cv=K, scoring="neg_log_loss")
    mean_acc = np.mean(acc)
    acc_arr = np.append(acc_arr, -mean_acc) # The sign is flipped in order to get the correct log loss

print(C_list)
print(acc_arr) #Smaller score is better
plt.figure()
plt.loglog(C_list,acc_arr)  
plt.show()

## E - Optimal value for C
opt_C = 1e-7
model = linear_model.LogisticRegression(C=opt_C).fit(X=X_train,y=y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))
pred_train = model.predict_proba(X_train)
pred_test = model.predict_proba(X_test)
print(metrics.log_loss(y_true=y_train, y_pred=pred_train))
print(metrics.log_loss(y_true=y_test, y_pred=pred_test))
