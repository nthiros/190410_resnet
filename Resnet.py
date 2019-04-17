import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets as kd

# Import Data
(x_train, y_train), (x_test, y_test) = kd.cifar10.load_data()
labels = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

x_train = x_train/255.
x_test = x_test/255.

# Convert class vectors to binary class matrices.
N = len(labels)

y_train = keras.utils.to_categorical(y_train, N)
y_test = keras.utils.to_categorical(y_test, N)


import tensorflow.keras.layers as kl
import tensorflow.keras.regularizers as kr

# Specify the shape of the input image
input_shape = x_train.shape[1:]
inputs = kl.Input(shape=input_shape)

# First convolution + BN + act
conv = kl.Conv2D(16,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(inputs)
bn = kl.BatchNormalization()(conv)
act1 = kl.Activation('relu')(bn)

# Perform 3 convolution blocks
for i in range(3):
    conv = kl.Conv2D(16,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act1)
    bn = kl.BatchNormalization()(conv)
    act = kl.Activation('relu')(bn)
    
    conv = kl.Conv2D(16,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
    bn = kl.BatchNormalization()(conv)

    # Skip layer addition
    skip = kl.add([act1,bn])
    act1 = kl.Activation('relu')(skip)  

# Downsampling with strided convolution
conv = kl.Conv2D(32,(3,3),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)
bn = kl.BatchNormalization()(conv)
act = kl.Activation('relu')(bn)

conv = kl.Conv2D(32,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
bn = kl.BatchNormalization()(conv)

# Downsampling with strided 1x1 convolution
act1_downsampled = kl.Conv2D(32,(1,1),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)

# Downsampling skip layer
skip_downsampled = kl.add([act1_downsampled,bn])
act1 = kl.Activation('relu')(skip_downsampled)

# This final layer is denoted by a star in the above figure
for i in range(2):
    conv = kl.Conv2D(32,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act1)
    bn = kl.BatchNormalization()(conv)
    act = kl.Activation('relu')(bn)
    
    conv = kl.Conv2D(32,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
    bn = kl.BatchNormalization()(conv)

    # Skip layer addition
    skip = kl.add([act1,bn])
    act1 = kl.Activation('relu')(skip)
    
# Downsampling with strided convolution
conv = kl.Conv2D(64,(3,3),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)
bn = kl.BatchNormalization()(conv)
act = kl.Activation('relu')(bn)

conv = kl.Conv2D(64,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
bn = kl.BatchNormalization()(conv)
act = kl.Activation('relu')(bn)

# Downsampling with strided 1x1 convolution
act1_downsampled = kl.Conv2D(64,(1,1),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)

# Downsampling skip layer
skip_downsampled = kl.add([act1_downsampled,bn])
act1 = kl.Activation('relu')(skip_downsampled)

# This final layer is denoted by a star in the above figure
for i in range(2):
    conv = kl.Conv2D(64,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act1)
    bn = kl.BatchNormalization()(conv)
    act = kl.Activation('relu')(bn)
    
    conv = kl.Conv2D(64,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
    bn = kl.BatchNormalization()(conv)

    # Skip layer addition
    skip = kl.add([act1,bn])
    act1 = kl.Activation('relu')(skip)


bn_final = kl.BatchNormalization()(act1)
gap = kl.GlobalAveragePooling2D()(bn_final)
final_bn = kl.BatchNormalization()(gap)
final_dense = kl.Dense(N)
fd = final_dense(final_bn)
softmax = kl.Activation('softmax')(fd)



import tensorflow.keras.models as km
import tensorflow.keras as keras
model = km.Model(inputs=inputs,outputs=softmax)

# initiate adam optimizer
opt = keras.optimizers.RMSprop(lr=0.001,decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


import tensorflow.keras.callbacks as kc
filepath = './checkpoints'

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = kc.ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 60:
        lr *= 1e-1
    elif epoch > 120:
        lr *= 1e-2
    return lr


lr_scheduler = kc.LearningRateScheduler(lr_schedule, verbose=1)

from keras.preprocessing.image import ImageDataGenerator
    
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.01,  # set range for random shear
    zoom_range=0.2,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

datagen.fit(x_train)


# We can include these two functions as *callbacks* to the optimizer:
batch_size = 64
epochs = 200

# Fit the model on the batches generated bydatagen.flow().
import os
f = 'output_model.h5'
if not os.path.isfile(f):
    model.fit_generator(datagen.flow(x_train, y_train,
                                    batch_size=batch_size),
                                    steps_per_epoch=len(x_train)/batch_size,
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),verbose=1, workers=4,
                                    callbacks=[checkpoint,lr_scheduler])
    model.save(f)
else:
    model.load_weights(f)


def visualize_CAM(model, data, label):
    import scipy.ndimage as snd
    new_model = km.Model(inputs=model.input,
            outputs=(bn_final, softmax))
    last_conv, probs = new_model.predict(data)
    pred = np.argmax(probs)
    title = "Pred: {}. True: {}".format(labels[pred], labels[label])
    weights = final_dense.get_weights()[0]
    weights = weights[:, pred]
    filters = last_conv[0]
    fm_0_upscaled = snd.zoom(filters[:, :, 0], 4)
    out = np.zeros_like(fm_0_upscaled)
    for i, weight in enumerate(weights):
        tmp = snd.zoom(filters[:, :, i], 4)
        out += snd.zoom(filters[:, :, i], 4)*abs(weight)
    fig, ax = plt.subplots(ncols=2)
    plt.suptitle(title)
    ax[0].imshow(data[0])
    ax[1].imshow(out,cmap=plt.cm.jet)
    plt.show()

rand = np.random.randint(0, len(x_train), size=10)
for idx in rand:
    data = np.expand_dims(x_train[idx], 0)
    label = np.argmax(y_train[idx])
    visualize_CAM(model, data, label)
