import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler # for scaling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score, recall_score, f1_score,classification_report
import matplotlib.pyplot as plt

#A1
def classification_metric(y_train,y_test,y_train_predict,y_test_predict):
    confusion=confusion_matrix(y_test,y_test_predict)
    precision_train=precision_score(y_train,y_train_predict)
    recall_train=recall_score(y_train,y_train_predict)
    f1_train=f1_score(y_train,y_train_predict)
    precision_test=precision_score(y_test,y_test_predict)
    recall_test=recall_score(y_test,y_test_predict)
    f1_test=f1_score(y_test,y_test_predict)
    return f"confusion matrix:{confusion},precision_train:{precision_train},recall_train:{recall_train},f1_train{recall_train},precision_test:{precision_test},recall_test:{recall_test},f1_test:{f1_test}"

#A2
def scores(y1_test,y1_predict):
    mse=np.mean((y1_test-y1_predict)**2)
    rmse=np.sqrt(mse)
    mape=np.mean(np.abs((y1_test-y1_predict)/y1_test)*100)
    r2_score=1-((np.sum((y1_test-y1_predict)**2))/(np.sum((y1_test-np.mean(y1_test))**2)))
    return f"mse:{mse},rmse:{rmse},mape:{mape},r^2 score:{r2_score}"

#A3
def assign_train_class():
    np.random.seed(42) # to give same random values every time we run
    X=np.random.randint(1,11,20)  
    Y=np.random.randint(1,11,20)
    labels=[]
    for x,y in zip(X,Y):
        if x+y>=9: # giving it as thresold i.e condition to assign class
            labels.append(1)
        else:
            labels.append(0)
    labels=np.array(labels)
    plt.figure(figsize=(6,6)) # 6 wide and 6 height
    for i in range(20):
        if labels[i]==0:
            plt.scatter(X[i],Y[i],color='blue',label='class 0' if i==0 else "") # if don't put "if i==0" then the legend prints class0 only for 20 times but legend is like showing the label and color for reference
        else:                                                                     # The legend only needs one example of each class, After that, repeating it is pointless
            plt.scatter(X[i],Y[i],color='red',label='class 1' if i==0 else "")
    plt.xlabel('X Feature')
    plt.ylabel('Y Feature')
    plt.grid(True)
    plt.legend()
    plt.show()
    return X,Y,labels
    
#A4
def assign_test_class(k):
    np.random.seed(42) # to give same random values every time we run
    X=np.arange(0,10,0.1)  
    Y=np.arange(0,10,0.1)
    xx,yy=np.meshgrid(X,Y) # for covering the entire 2D space
    X_train,Y_train,train_labels=assign_train_class()
    train_points = [[x, y] for x, y in zip(X_train, Y_train)]
    test_points  = [[x, y] for x, y in zip(xx.ravel(), yy.ravel())] # ravel flattens to 1D because scatter can't take 2D
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_points,train_labels)
    test_labels=neigh.predict(test_points)
    plt.figure(figsize=(6,6))
    colors=['blue' if label==0 else 'red' for label in test_labels]
    plt.scatter(
        xx.ravel(),
        yy.ravel(),
        c=colors
    )
    plt.xlabel('X Feature')
    plt.ylabel('Y Feature')
    plt.grid(True)
    plt.show()

#A5
def different_k():
    for k in [1,5,7]:
        assign_test_class(k)
        
#A6
def A3(X2_train,y2_train):
    plt.figure(figsize=(6,6)) # 6 wide and 6 height
    plt.scatter(X2_train[y2_train==0,0],X2_train[y2_train==0,1],color='blue',label='class 0') # if don't put "if i==0" then the legend prints class0 only for 20 times but legend is like showing the label and color for reference                                                                    # The legend only needs one example of each class, After that, repeating it is pointless
    plt.scatter(X2_train[y2_train==1,0],X2_train[y2_train==1,1],color='red',label='class 1')
    plt.xlabel('X Feature')
    plt.ylabel('Y Feature')
    plt.grid(True)
    plt.title("Dataset training")
    plt.show()
    
def A4(X2_train,X2_test,y2_train,k):
    neigh=KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X2_train,y2_train)
    dataset_predict=neigh.predict(X2_test)
    plt.figure(figsize=(6,6))
    colors=['blue' if label==0 else 'red' for label in dataset_predict]
    plt.scatter(
        X2_test[:,0],
        X2_test[:,1],
        c=colors
        )
    plt.xlabel('X Feature')
    plt.ylabel('Y Feature')
    plt.grid(True)
    plt.title("Dataset testing")
    plt.show()
    
def A5():
    for k in [1,5,7]:
        A4(X2_train,X2_test,y2_train,k)
        
#A7
def best_k(X2_train,y2_train,X2_test,y2_test):
    parameter={
        "n_neighbors":[1,3,5,7,9,11,13]
    }
    knn = KNeighborsClassifier()
    grid=GridSearchCV(
        estimator=knn,
        param_grid=parameter,
        cv=5, # cross validation with 5 folds, 4 as training as 1 as testing
        scoring='accuracy'
    )
    grid.fit(X2_train, y2_train)
    print("Best k value:", grid.best_params_['n_neighbors'])
    print("Best cross-validation accuracy:", grid.best_score_)
    best_knn = grid.best_estimator_
    y_pred = best_knn.predict(X2_test)
    return f"accuracy_score:{accuracy_score(y2_test, y_pred)},Confusion Matrix:{confusion_matrix(y2_test, y_pred)},Classification Report:{classification_report(y2_test, y_pred)}"



if __name__ == "__main__":
    
    #2
    data1=pd.read_excel("Lab02.xlsx",sheet_name="IRCTC_Stock_Price")
    X1=data1.drop(columns=['Date','Month','Day','Volume','Chg%']).values
    y1=data1["Chg%"].values
    X1_train,X1_test,y1_train,y1_test=train_test_split(X1,y1,test_size=0.3)
    scaler=StandardScaler()
    X1_train=scaler.fit_transform(X1_train)
    X1_test=scaler.transform(X1_test)
    neigh1=KNeighborsRegressor(n_neighbors=3)
    neigh1.fit(X1_train,y1_train)
    y1_predict=neigh1.predict(X1_test)
    print(scores(y1_test,y1_predict))
    
    #1
    data=pd.read_excel("Lab02.xlsx",sheet_name="marketing_campaign")
    data=pd.get_dummies(data,drop_first=True) # converting categorical columns into numeric
    data=data.fillna(data.mean()) # fill NaN values with mean
    X=data.drop("Response",axis=1).values
    y=data["Response"].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    neigh=KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train,y_train)
    y_train_predict=neigh.predict(X_train)
    y_test_predict=neigh.predict(X_test)
    print(classification_metric(y_train,y_test,y_train_predict,y_test_predict))
    
    print(assign_train_class())
    print(assign_test_class(3))
    print(different_k)
    
    #A6
    data2=pd.read_excel("Lab02.xlsx",sheet_name="marketing_campaign")
    data2=data2[['Income','MntWines','Response']]
    data2=data2.fillna(data2.mean())
    X2=data2[['Income','MntWines']].values
    y2=data2['Response'].values
    X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.3)
    scaler=StandardScaler()
    X2_train=scaler.fit_transform(X2_train)
    X2_test=scaler.transform(X2_test)
    print(A3(X2_train,y2_train))
    print(A4(X2_train,X2_test,y2_train,3))
    print(A5())
    
    print(best_k(X2_train,y2_train,X2_test,y2_test))
