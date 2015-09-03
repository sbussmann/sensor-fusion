"""

Take a processed dataframe, generate features, add classification label, make
validation set, predict class.

"""

import pandas as pd
from scipy.ndimage import gaussian_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


df = pd.read_csv('shaneiphone_exp2_processed.csv', index_col='DateTime')

# We are allowed to use only userAcceleration and gyroscope data
xyz = ['X', 'Y', 'Z']
measures = ['userAcceleration', 'gyroscope']
features = [i + j for i in measures for j in xyz]

# Generate Gaussian smoothed features
smoothfeatures = []
for i in features:
    df[i + 'sm'] = gaussian_filter(df[i], 3)
    df[i + '2sm'] = gaussian_filter(df[i], 50)
    smoothfeatures.append(i + 'sm')
    smoothfeatures.append(i + '2sm')
features.extend(smoothfeatures)

# Generate Fourier Transform of features
fftfeatures = []
for i in features:
    reals = np.real(np.fft.rfft(df[i]))
    imags = np.imag(np.fft.rfft(df[i]))
    complexs = [reals[0]]
    for j in range(1, reals.size - 1):
        complexs.append(reals[j])
        complexs.append(imags[j])
    complexs.append(reals[j])
    n = len(reals)
    if n % 2 != 0:
        complexs.append(imags[j])
    df['f' + i] = complexs
    fftfeatures.append('f' + i)
features.extend(fftfeatures)

df = df[features]

# assign class labels
class0 = (df.index > '2015-08-25 14:35:00') & \
        (df.index < '2015-08-25 14:42:00')

class1 = (df.index > '2015-08-25 14:43:00') & \
        (df.index < '2015-08-25 14:48:00')

df['class'] = -1
df['class'][class0] = 0
df['class'][class1] = 1

# remove the unclassified portion of the data
classed = df['class'] != -1
df = df[classed]

# separate into quarters for train and validation
q1 = df[(df.index <= '2015-08-25 14:38:30') & 
        (df.index > '2015-08-25 14:33:00')]
q2 = df[(df.index > '2015-08-25 14:38:30') & 
        (df.index <= '2015-08-25 14:42:00')]
q3 = df[(df.index > '2015-08-25 14:43:00') & 
        (df.index <= '2015-08-25 14:45:30')]
q4 = df[(df.index > '2015-08-25 14:45:30') & 
        (df.index <= '2015-08-25 14:48:00')]
traindf = pd.concat([q1, q3])
validationdf = pd.concat([q2, q4])

# drop NaNs
traindf = traindf.dropna()
validationdf = validationdf.dropna()
X_train = traindf[traindf.columns[0:-1]].values
Y_train = traindf[traindf.columns[-1]].values
X_test = validationdf[validationdf.columns[0:-1]].values
Y_test = validationdf[validationdf.columns[-1]].values

# train a random forest
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)
scores = cross_val_score(clf, X_train, Y_train, cv=5)
Y_test_RFC = clf.predict(X_test)

print("Results from cross-validation on training set:")
print(scores, scores.mean(), scores.std())

testscore = accuracy_score(Y_test, Y_test_RFC)

print("Accuracy score on test set: %6.3f" % testscore)

print("Feature importances: ")
for i, ifeature in enumerate(features):
    print(ifeature + ': %6.4f' % clf.feature_importances_[i])

import pdb; pdb.set_trace()
