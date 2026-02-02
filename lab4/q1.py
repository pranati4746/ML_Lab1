import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'Candies':[10,20,12,22,15,30,18,25],
    'Mangoes':[2,6,1,5,3,4,2,7],
    'MilkPackets':[3,4,2,3,2,4,1,3],
    'Payment':[250,480,190,410,270,520,230,500]
}

df = pd.DataFrame(data)
df['Class'] = df['Payment'].apply(lambda x: 'High' if x>=300 else 'Low')

X = df[['Candies','Mangoes','MilkPackets']]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print("Train Confusion Matrix:")
print(confusion_matrix(y_train, train_pred))

print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nTraining Metrics:")
print(classification_report(y_train, train_pred))

print("\nTesting Metrics:")
print(classification_report(y_test, test_pred))
