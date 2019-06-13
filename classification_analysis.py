# Loading data using sklearn dataset
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, ShuffleSplit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

feature_names = np.append(cancer.feature_names, ['target'])
# 0 Malignant, 1 Benign (0, 19)
print(np.unique(y))
print(f'sklearn cancer dataset X shape: {X.shape}')
print(f'sklearn cancer dataset y shape: {y.shape}')
print(f'Features names {feature_names}')
print(f'Target name {cancer.target_names}')
print(f'keys: {cancer.keys()}')

tmp = np.c_[cancer.data, cancer.target]
cancer_df = pd.DataFrame(tmp, columns=feature_names)
print(cancer_df.head())

# ************************************
# ********** Applying PCA ************
# ************************************
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print(X_scaled)

pca = PCA(n_components=30)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

print('shape of X_pca', X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print('sum  8: ', sum(expl[0:8]))
print('sum 10: ', sum(expl[0:10]))
print('sum 12: ', sum(expl[0:12]))
print('sum 15: ', sum(expl[0:15]))
print('sum 20: ', sum(expl[0:20]))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

Xax = X_pca[:, 0]
Yax = X_pca[:, 1]
labels = cancer_df['target'].values
cdict = {0: 'red', 1: 'green'}
labl = {0: 'Malignant', 1: 'Benign'}
marker = {0: 'o', 1: '*'}
alpha = {0: .3, 1: .5}
fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
    ix = np.where(labels == l)
    ax.scatter(Xax[ix], Yax[ix], c=cdict[l], label=labl[l], s=40, marker=marker[l], alpha=alpha[l])

plt.xlabel("First Principal Component", fontsize=14)
plt.ylabel("Second Principal Component", fontsize=14)
plt.legend()
plt.show()

# ************************************
# ********** Split dataset ***********
# ************************************

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.35)

# Printing splitted datasets
print(f'X_train.shape : {X_train.shape}, y_train.shape : {y_train.shape}')
print(f'X_test.shape : {X_test.shape}, y_test.shape : {y_test.shape}')

# ************************************
# ********** Training model***********
# ************************************

lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=4000)
lr.fit(X_train, y_train)

print(f'Intercept per class: {lr.intercept_}\n')
print(f'Coeficients per class: {lr.coef_}\n')
print(f'Available classes : {lr.classes_}\n')
print(f'Named Coeficients for class 0: {pd.DataFrame(lr.coef_[0], cancer.feature_names)}\n')
print(f'Number of iterations generating model : {lr.n_iter_}')

# from sklearn.externals import joblib
# joblib.dump(lr, 'model.pkl')

# ************************************
# ********** Predicting the results **
# ************************************

predicted_values = lr.predict(X_test)

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f'Value: {real}, pred:{predicted} {"is different " if real != predicted else ""}')

# ************************************
# ********** Accuracy SCore **********
# ************************************
print(f'Accuracy score is {lr.score(X_test, y_test):.2f}/1 \n')

# ************************************
# ******** Printing Reports **********
# ************************************
print('Classification report\n')
print(classification_report(y_test, predicted_values))
print('Confusion matrix\n')
print(confusion_matrix(y_test, predicted_values))
print('Overal f1-score\n')
print(f1_score(y_test, predicted_values, average="macro"))


# ************************************
# ******** Cross Validation **********
# ************************************
print(f'Cross_val_score before ShuffleSplit')
print(cross_val_score(lr, X, y, cv=10))
cv = ShuffleSplit(n_splits=5)
print(f'Cross_val_score after ShuffleSplit')
print(cross_val_score(lr, X, y, cv=cv))
