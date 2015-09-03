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


dfcar = pd.read_csv('shaneiphone_exp2_processed.csv', index_col='DateTime')
dfbus = pd.read_csv('shanebus20150827_processed.csv', index_col='DateTime')

df = pd.concat([dfcar, dfbus])

# We are allowed to use only userAcceleration and gyroscope data
xyz = ['X', 'Y', 'Z']
measures = ['userAcceleration', 'gyroscope']
features = [i + j for i in measures for j in xyz]

# Generate Gaussian smoothed features
smoothfeatures = []
for i in features:
    df[i + 'sm'] = gaussian_filter(df[i], 3)
    df[i + '2sm'] = gaussian_filter(df[i], 100)
    smoothfeatures.append(i + 'sm')
    smoothfeatures.append(i + '2sm')
features.extend(smoothfeatures)

# Generate Jerk signal
jerkfeatures = []
for i in features:
    diffsignal = np.diff(df[i])
    df[i + 'jerk'] = np.append(0, diffsignal)
    jerkfeatures.append(i + 'jerk')
features.extend(jerkfeatures)

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
    df['f' + i] = complexs
    fftfeatures.append('f' + i)
features.extend(fftfeatures)
print(features)

df = df[features]

# assign class labels
car0 = (df.index > '2015-08-25 14:35:00') & \
        (df.index <= '2015-08-25 14:42:00')

car1 = (df.index > '2015-08-25 14:43:00') & \
        (df.index <= '2015-08-25 14:48:00')

bus0 = (df.index > '2015-08-27 10:10:00') & \
        (df.index <= '2015-08-27 10:15:00')
bus1 = (df.index > '2015-08-27 10:15:00') & \
        (df.index <= '2015-08-27 10:20:00')

nc = len(df)
df['class'] = np.zeros(nc) - 1
df['class'][car0] = np.zeros(nc)
df['class'][car1] = 0
df['class'][bus0] = 1
df['class'][bus1] = 1

## remove the unclassified portion of the data
#classed = df['class'] != -1
#df = df[classed]

# separate into quarters for train and validation
q1 = df[car0]
q2 = df[car1]
q3 = df[bus0]
q4 = df[bus1]
traindf = pd.concat([q2, q4])
validationdf = pd.concat([q1, q3])

# drop NaNs
traindf = traindf.dropna()
validationdf = validationdf.dropna()
X_train = traindf[traindf.columns[0:-1]].values
Y_train = traindf[traindf.columns[-1]].values
X_test = validationdf[validationdf.columns[0:-1]].values
Y_test = validationdf[validationdf.columns[-1]].values

# train a random forest
print("Beginning random forest classification")
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, Y_train)
scores = cross_val_score(clf, X_train, Y_train, cv=5)
Y_test_RFC = clf.predict(X_test)

print("Results from cross-validation on training set:")
print(scores, scores.mean(), scores.std())

testscore = accuracy_score(Y_test, Y_test_RFC)

print("Accuracy score on test set: %6.3f" % testscore)

print("Feature importances:")
print(clf.feature_importances_)

import pdb; pdb.set_trace()
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
#from keras.layers.embeddings import Embedding
#from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath="testnn.hdf5", verbose=1, save_best_only=True)
model = Sequential()
# Add a mask_zero=True to the Embedding connstructor if 0 is a left-padding value in your data
max_features = 200000
model.add(Dense(X_train.shape[1], 256))
model.add(Activation('relu'))
#model.add(Embedding(X_train.shape[1], 256))
#model.add(LSTM(256, 128, activation='sigmoid', inner_activation='hard_sigmoid'))
#model.add(Dropout(0.5))
model.add(Dense(256, 128))
model.add(Activation('relu'))
model.add(Dense(128, 1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')


call_back = model.fit(X_train, Y_train, batch_size=256, nb_epoch=200, 
                      show_accuracy=True, validation_data=[X_test, Y_test], 
                      callbacks=[checkpointer])

#score = model.evaluate(X_test, Y_test, batch_size=256)


# In[430]:

#plt.plot(model.predict(X_test))
import pdb; pdb.set_trace()
