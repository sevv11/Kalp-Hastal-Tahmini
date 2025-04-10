
#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")



#load dataset and eda 

df = pd.read_csv("heart_disease_uci.csv")

df = df.drop(columns = ["id"])

df.info()

describe = df.describe()

numerical_features = df.select_dtypes(include =[np.number]).columns.tolist()

plt.figure()
sns.pairplot(df,vars = numerical_features, hue = "num")
plt.show()

plt.figure()
sns.countplot(x="num", data = df)
plt.show()

#handling missing value
df.isnull()

#false değerleri kayıp veri olmadığı anlamına gelir.


df.isnull().sum()

#611 null değeri var

print(df.isnull().sum())

df= df.drop(columns = ["ca"])

print(df.isnull().sum())

#ca dan kurtulmuş olduk
# thal ve slope feature larını medyan değerlerine göre dolduracağız.

df["trestbps"].fillna(df["trestbps"].median() , inplace = True)
df["chol"].fillna(df["chol"].median() , inplace = True)
df["fbs"].fillna(df["fbs"].mode()[0] , inplace = True)
df["restecg"].fillna(df["restecg"].mode()[0] , inplace = True)
df["thalch"].fillna(df["thalch"].median() , inplace = True)
df["exang"].fillna(df["exang"].mode()[0] , inplace = True)
df["oldpeak"].fillna(df["oldpeak"].median() , inplace = True)
df["slope"].fillna(df["slope"].mode()[0] , inplace = True)
df["thal"].fillna(df["thal"].mode()[0] , inplace = True)

print(df.isnull().sum())

# veri setinden çıkarmak
# numericleri medyana kategorikleri moda göre doldurma yapmak
# medyan yerine ortalama değerleri de alabiliriz

#train test split
x = df.drop(["num"], axis = 1)

y = df["num"]


x_train , x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# 920 adet verinin 690 tanesi train 230 tanesi de test olarak ayrıldı.

categorical_features = ["sex","dataset","cp","restecg","exang","slope","thal"]
numerical_features = ["age","trestbps","chol","fbs","thalch","oldpeak"]


#standartizasyon 


x_train_num = x_train[numerical_features]

x_test_num = x_test[numerical_features]

scaler = StandardScaler()
x_train_num_scaled = scaler.fit_transform(x_train_num)
x_test_num_scaled = scaler.fit_transform(x_test_num)

#categoric verilerde scaled işlemi yoktur.

#encoder işlemi ile kodlamamız lazım
#kategorig kodlama 
encoder=OneHotEncoder(sparse_output=False, drop="first")

x_train_cat = x_train[categorical_features]
x_test_cat = x_test[categorical_features]

x_train_cat_encoded = encoder.fit_transform(x_train_cat)
x_test_cat_encoded = encoder.fit_transform(x_test_cat)


#categoric degerlerin encoded hali
#numeric degerlerin scaled hali

#bir veri setinde birleştirmem lazım 

x_train_transformed = np.hstack((x_train_num_scaled, x_train_cat_encoded))
x_test_transformed = np.hstack((x_test_num_scaled, x_test_cat_encoded))

#modeling: rf knn voting classifier train ve test

rf= RandomForestClassifier(n_estimators=100, random_state=42)
knn = KNeighborsClassifier()

voting_clf = VotingClassifier(estimators=[
    ("rf", rf),
    ("knn",knn)],voting="soft")

#model eğitimi

voting_clf.fit(x_train_transformed, y_train)

#test verisi ile tahmin yap

y_pred = voting_clf.predict(x_test_transformed)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))


#cm

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

































