import numpy as np
import cv2
import pandas as pd
import sys
import typing
from sklearn.metrics import fbeta_score
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.models import Model

np.random.seed(42)


assert len(sys.argv) == 2, 'Specify the type of algorithm by 0 or 1'
assert sys.argv[1] in [str(0),str(1)], '2nd argument must be either 1 for pre-trained VGG or 0 for a model from scratch'


# Build a pretrained VGG with imagenet weights, without the top part.
def pretrained_vgg(size_image: float) -> Model:
    input_shape = (size_image, size_image, 3)
    model_ = VGG16(include_top=False,
                   weights='imagenet',
                   input_shape=input_shape)
    return model_


# Add the top part of the VGG 16 in order to classify our samples, in order
# to fit the parameters of the dense layers onto our data
def complete_vgg(size_image: float) -> Model:
    base_network = pretrained_vgg(size_image)
    input_ = Input(shape=(size_image, size_image, 3))
    preprocessed = base_network(input_)
    x = Flatten(name='flatten')(preprocessed)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions_ = Dense(17, activation='sigmoid')(x)
    model_ = Model(input=input_, outputs=predictions_)
    return model_
# Optimization of the thresholds on the validation set


def optimise_f2_thresholds(y: typing.Iterable[float],
                           p: typing.Iterable[float],
                           verbose=True,
                           resolution=100) -> typing.Iterable[float]:

    def mf(x: typing.Iterable[float]) -> float:
        p2 = np.zeros_like(p)
        for i in range(17):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score_ = fbeta_score(y, p2, beta=2, average='samples')
        return score_

    x = [0.2]*17
    for i in range(17):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score_ = mf(x)
            if score_ > best_score:
                best_i2 = i2
                best_score = score_
        x[i] = best_i2
        if verbose:
            print(i, best_i2, best_score)
    return x

# Simulation
# sys.argv should be length 2, second argument must be 1 or 0 depending on which model
# is chosen for the fitting



# shape of the image are (size,size)
size = 48
x_train = []
x_test = []
y_train = []

# Loading the data
df_train = pd.read_csv('Data/train_v2.csv')
flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))
label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('Data/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    if img is not None:
        x_train.append(cv2.resize(img, (size, size)))
        y_train.append(targets)

    
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255

# Splitting The data - Must be improved by doing k fold cv
split = 35000
x_train, x_valid, y_train, y_valid = x_train[:split], x_train[split:], y_train[:split], y_train[split:]


if sys.argv[1] == '0':
    print('Building a model from scratch')

    # Small version of the VGG 16
    model = Sequential()

    # block 1
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(size, size, 3)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # block 2
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense layers
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))

    # Dense with 17 layers - each neuron corresponding to a class
    # Indeed each image can contain between 0 and all the labels.
    # The output vector then contain the probability of having the
    # each of the possible labels.
    model.add(Dense(17, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.0005),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=20,
              verbose=1,
              validation_data=(x_valid, y_valid))
else:
    print('Training the top of the VGG 16')
    model = complete_vgg(size)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.0005),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=11,
              verbose=1,
              validation_data=(x_valid, y_valid))

p_valid = model.predict(x_valid, batch_size=128)
print('Searching for optimal tresholds of Beta score on the validation set')
thresholds = optimise_f2_thresholds(y_valid, p_valid)


# Predicting Submission Data
print('Preparing Submission Data')
test = pd.read_csv('Data/sample_submission_v2.csv')

x_sub = []
for f, tags in tqdm(test.values, miniters=1000):
    img = cv2.imread('Data/test-jpg/{}.jpg'.format(f))
    if img is not None:
        x_sub.append(cv2.resize(img, (size, size)))
x_sub = np.array(x_sub, np.float16) / 255.

pred = model.predict(x_sub).astype(float)
pred = pred > thresholds

predictions = []
for i in range(pred.shape[0]):
    labs = [inv_label_map[k] for k in range(pred[i, :].shape[0]) if pred[i, k]]
    predictions.append(' '.join(labs))
test['tags'] = np.array(predictions)
test.to_csv('submission.csv', index=False)
