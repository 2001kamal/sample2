import pandas as pd
df = pd.read_csv("/content/BMI.csv")
df.head()
df.shape
df.info()
df.isna().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform (df['Gender'])
print(df.head())
x = df[['Gender', 'Height', 'Weight']]
y = df.Index
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(xtrain, ytrain)
ypred = knn.predict(xtest)
print(ypred)
from sklearn.metrics import classification_report, accuracy_score
print(accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))
