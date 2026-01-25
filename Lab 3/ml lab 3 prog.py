import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import mne
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# A1
def manual_dot(A,B):
    dot_pro=0
    for i in range(len(A)): 
        dot_pro=dot_pro+A[i]*B[i] # dot product
    return f"Manual Dot product:{dot_pro}"
    
def manual_norm_A(A):
    total_A=0
    for i in range(len(A)):
        total_A=total_A+(A[i]*A[i]) # norm
    return f"manual norm of A:{math.sqrt(total_A)}"

def manual_norm_B(B):
    total_B=0
    for i in range(len(B)):
        total_B=total_B+(B[i]*B[i]) # norm
    return f"manual norm of B:{math.sqrt(total_B)}"

def package_dot(A,B):
    return f"Package dot product:{np.dot(A,B)}" # inbuilt function dot

def package_norm(A,B):
    return f"Package norm of A and B:{np.linalg.norm(A), np.linalg.norm(B)}" # inbuilt function norm

#A2
def calculate_mean(A):
    return np.mean(A)

def calculate_variance(A):
    return np.var(A)

def calculate_std(A):
    return np.std(A)

def calculate_data_set(X):
    mean_dataset=calculate_mean(X) # mean of dataset
    var_dataset = calculate_variance(X) # var of dataset
    std_dataset = calculate_std(X) # std of dataset
    return mean_dataset, var_dataset, std_dataset

def class_mean(X,y): #class centroid
    centroid1=np.mean(X[y==0],axis=0) #filtering the class 0 (AD)
    centroid2=np.mean(X[y==2],axis=0) #filtering the class 2 (CN)
    return f"centroid1:{centroid1}, centroid:{centroid2}"

def class_std(X,y): # spread
    spread1=np.std(X[y==0],axis=0) #filtering the class 0 (AD)
    spread2=np.std(X[y==2],axis=0) #filtering the class 2 (CN)
    return f"spread1:{spread1}, spread2:{spread2}"

def distance(X,y):
    centroid1=np.mean(X[y==0],axis=0) # mean of class AD
    centroid2=np.mean(X[y==2],axis=0) # mean of class CN
    dist=np.linalg.norm(centroid1-centroid2) # norm of means
    return f"distance:{dist}"

#A3
def plot_hist(X):
    feature_vec=X[:,0]  #extracting 1st feature from the dataset (1st column)
    feature_vec_mean=np.mean(feature_vec) #mean of feature vector
    feature_vec_var=np.var(feature_vec) #variance of feature vector
    hist_values, bin_edges = np.histogram(feature_vec, bins=10)
    plt.hist(feature_vec, bins=10)
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Selected Feature")
    plt.show()
    return feature_vec_mean,feature_vec_var

#A4
def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2) ** p) ** (1 / p) # minkowski distance

def hist_distance_plot(x1,x2): # plotting histogram for a range of values
    p_values=range(1,11)
    distances = []
    for p in p_values: 
       d = minkowski_distance(x1, x2, p)
       distances.append(d)
    plt.plot(p_values, distances, marker='o')
    plt.xlabel("p value")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance for p = 1 to 10")
    plt.show()

#A5
def compare_minkowski_dist(x1,x2):
    for p in range(1,11):
       package_minkowski=minkowski(x1,x2,p) # inbuilt minkowski distance function
       manual_min=minkowski_distance(x1,x2,p) # minkowski distance calculated
       return f"manual minkowski:{manual_min},package_minkowski:{package_minkowski}"

#A7
def knn_class(train_X,train_y):
    k=KNeighborsClassifier(n_neighbors=3) #KNN algorithm for k=3, search for 3 nearest points but didn't yet give data to it. A fter giving data it starts searching
    k.fit(train_X,train_y) #gave data for searching
    return k

#A8
def accuracy(test_X,test_y):
    k1=knn_class(train_X,train_y)
    accuracy=k1.score(test_X,test_y) #generates an accuracy report of the KNN by giving test data to it. so that how much accurately it predicted
    return f"Accuracy:{accuracy}"

#A9
def predict_neigh(test_X):
    k_data=knn_class(train_X,train_y)
    test_X_vec=test_X[0].reshape(1,-1) #taking 1 subject(1 pateint's data) which is a vector of 19 features, reshape is taking 1 row and -1 indicates take all columns
    pre=k_data.predict(test_X_vec) #produces class(0 or 2) of the test vector
    return f"Predict:{pre}"

#A10
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def my_knn_classifier(X_train, y_train, new_point, k):
    distances = []
    for i in range(len(X_train)):
        d = euclidean_distance(X_train[i], new_point) # calculating distance between data and new data point
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0]) # sorting those distances for finding k nearest neighbors
    k_nearest = distances[:k] # extracting 1st k from the distance array
    labels = [label for _, label in k_nearest] 
    return max(set(labels), key=labels.count) # taking the maximum occuring class

def compare(train_X,test_X,train_y,test_y):
    k = 3
    y_predict_myknn=my_knn_classifier(train_X, train_y, test_X[0], k) #taking 1 subject(1 pateint's data) which is a vector of 19 features
    predict_package=predict_neigh(test_X)
    return f"K nearest neighbor_manual:{y_predict_myknn}, package_K nearest neighbor:{predict_package}"

#A11
def compare_k1_k3(train_X,train_y,test_X,test_y):
    nn = KNeighborsClassifier(n_neighbors=1) # KNN for k=1
    nn.fit(train_X, train_y) # training
    acc_nn = nn.score(test_X,test_y) # accuracy of k=1
    knn = KNeighborsClassifier(n_neighbors=3) # KNN for k=3
    knn.fit(train_X,train_y) # training
    acc_knn = knn.score(test_X,test_y) # accuracy of k=3
    return f"Accuracy (k=1):{acc_nn},Accuracy (k=3):{acc_knn}"

def plot_knn_for_kvalues(train_X,train_y,test_X,test_y):
    k_values = range(1, 12)
    accuracies = []
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k) # KNN for range of values of K
        model.fit(train_X,train_y)
        accuracies.append(model.score(test_X,test_y))
    plt.plot(k_values, accuracies, marker='o') # ploting the accuracies for each k
    plt.xlabel("k value")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k (1 to 11)")
    plt.show()
    
#A12
def performance_metrics(train_X,train_y,test_X,test_y):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_X,train_y)
    y_train_pred = knn.predict(train_X) # predicting
    y_test_pred = knn.predict(test_X)
    cm_train = confusion_matrix(train_y, y_train_pred) # confusion matrix
    cm_test = confusion_matrix(test_y, y_test_pred)
    return f"Accuracy :{accuracy_score(train_y,y_train_pred)},Precision:{precision_score(train_y, y_train_pred,pos_label=0)},Recall:{recall_score(train_y, y_train_pred,pos_label=0)},F1-Score:{f1_score(train_y, y_train_pred,pos_label=0)}"

#A13
def confusion_matrix_manual(train_X,train_y,test_X,test_y):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_X,train_y)
    y_pred = knn.predict(test_X)
    TP = FP = FN = TN = 0
    for i in range(len(test_y)):
        if test_y[i] == 2 and y_pred[i] == 2:
            TP += 1
        elif test_y[i] == 0 and y_pred[i] == 2:
            FP += 1
        elif test_y[i] == 2 and y_pred[i] == 0:
            FN += 1
        else:
            TN += 1
    return TP, FP, FN, TN

def accuracy_score_manual(TP, FP, FN, TN):
    return (TP + TN) / (TP + FP + FN + TN)

def precision_score_manual(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall_score_manual(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def fbeta_score_manual(TP, FP, FN, beta=1):
    precision = precision_score_manual(TP, FP)
    recall = recall_score_manual(TP, FN)
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

def compare_performance(train_X,train_y,test_X,test_y):
    TP, FP, FN, TN = confusion_matrix_manual(train_X,train_y,test_X,test_y)
    accuracy = accuracy_score_manual(TP, FP, FN, TN)
    precision = precision_score_manual(TP, FP)
    recall = recall_score_manual(TP, FN)
    f1_score = fbeta_score_manual(TP, FP, FN, beta=1)
    return accuracy,precision,recall,f1_score

#A14

if __name__ == "__main__":
    
    # data extraction
 participants = pd.read_csv(
    r"D:\ML\ds004504\ds004504\participants.tsv", 
    sep="\t"
    )

 group_map = {
    "A": 0,   # AD
    "F": 1,   # FTD
    "C": 2    # CN
}

 label_dict = {}

 for _, row in participants.iterrows():
    subject_id = row["participant_id"]
    group_code = row["Group"]          # numeric
    label_dict[subject_id] = group_map[group_code]
    
 base_path = r"D:\ML\ds004504\ds004504\derivatives"

 X = []
 y = []

 for subject_id in label_dict.keys():

    eeg_dir = os.path.join(base_path, subject_id, "eeg")

    if not os.path.exists(eeg_dir):
        print("Missing EEG dir:", eeg_dir)
        continue

    set_files = [f for f in os.listdir(eeg_dir) if f.endswith(".set")]

    if len(set_files) == 0:
        print("No .set file in:", eeg_dir)
        continue

    eeg_file = os.path.join(eeg_dir, set_files[0])

    raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
    data = raw.get_data()

    features = np.mean(data, axis=1)
    X.append(features)
    y.append(label_dict[subject_id])
 X = np.array(X)   # converting list into numpy arrays
 y = np.array(y)

 filter_2 = (y == 0) | (y == 2)
 X = X[filter_2]   # filtering to 2 classes because dataset is multiclass
 y = y[filter_2]
 
 #A6
 train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.3) #test_size is the spliting the test data into 30%   
 
 A=np.array([6,7,4,8,1])
 B=np.array([9,0,3,2,5])
 print(manual_dot(A,B))
 print(manual_norm_A(A))
 print(manual_norm_B(B))
 print(package_dot(A,B))
 print(package_norm(A,B))
 print(calculate_data_set(X))
 print(class_mean(X,y))
 print(class_std(X,y))
 print(distance(X,y))
 print(plot_hist(X))
 x1=X[0] # 1st feature vector(1st subject)
 x2=X[1] # 2nd feature vector(2nd subject)
 print(hist_distance_plot(x1,x2))
 print(compare_minkowski_dist(x1,x2))
 print(accuracy(test_X,test_y))
 print(predict_neigh(test_X))
 print(compare(train_X,test_X,train_y,test_y))
 print(compare_k1_k3(train_X,train_y,test_X,test_y))
 print(plot_knn_for_kvalues(train_X,train_y,test_X,test_y))
 print(performance_metrics(train_X,train_y,test_X,test_y))
 print(compare_performance(train_X,train_y,test_X,test_y))
