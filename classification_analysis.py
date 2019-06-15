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
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('figures', exist_ok=True)

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

feature_names = np.append(cancer.feature_names, ['target'])

print(np.unique(y))
print(f'sklearn cancer dataset X shape: {X.shape}')
print(f'sklearn cancer dataset y shape: {y.shape}')
print(f'Features names {feature_names}')
print(f'Target name {cancer.target_names}')
print(f'keys: {cancer.keys()}')

tmp = np.c_[cancer.data, cancer.target]
cancer_df = pd.DataFrame(tmp, columns=feature_names)

# Note that we need to reverse the original '0' and '1' mapping in order to end up with this mapping:
# Benign = 0 (negative class)
# Malignant = 1 (positive class)
cancer_df['target'] = cancer_df['target'].apply(lambda x: 0 if x == 1 else 1)
print(cancer_df.head())
print(cancer_df.iloc[:,:6].describe())

CONST_B = 0; CONST_M = 1
cancer_df['diagnosis_ds'] = cancer_df['target'].map({CONST_B: 'Benign', CONST_M: 'Malignant'})

######################################
# ### Visualization
######################################

# count of obs in each class
benign, malignant = cancer_df['target'].value_counts()
print('Number of cells class 0 - Benign:', benign)
print('Number of cells class 1- Malignant : ', malignant)
print('')
print('% of cells class 0 - Benign', round(benign / len(cancer_df) * 100, 2), '%')
print('% of cells class 1 - Malignant', round(malignant / len(cancer_df) * 100, 2), '%')

# Countplot
plt.figure(figsize=(6, 3))
sns.countplot(cancer_df['target'])
plt.savefig('figures/countplot.png')
plt.close()

# Heatmap
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cancer_df.corr(), vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
ax.set_xticklabels(cancer_df.columns, rotation=45, horizontalalignment='right')
ax.set_yticklabels(cancer_df.columns, rotation=45)
fig.tight_layout()
plt.savefig('figures/heatmap-all.png')
plt.close()


#####################################
# ### Feature Reduction
#####################################
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# n_components = 4
pca = PCA(n_components=4)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

print('shape of X_pca', X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print('sum of 2 components: ', sum(expl[0:2]))
print('sum of 3 components: ', sum(expl[0:3]))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('figures/pca-plot-n-4.png')
plt.close()

# n_components = 3
pca = PCA(n_components=3)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

print('shape of X_pca', X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print('sum of 2 components: ', sum(expl[0:2]))
print('sum of 3 components: ', sum(expl[0:3]))

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.savefig('figures/pca-plot-n-3.png')
plt.close()

Xax = X_pca[:, 0]
Yax = X_pca[:, 1]
labels = cancer_df['target'].values
cdict = {0: 'red', 1: 'green'}
labl = {0: 'Benign', 1: 'Malignant'}
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
plt.savefig('figures/pca-scatter-n-3.png')
plt.close()


# ************************************
# ********** Split the data ***********
# ************************************
TEST_SIZE_RATIO = 0.35
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE_RATIO)

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
