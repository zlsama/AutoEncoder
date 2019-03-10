from keras.layers import Dense,Input
from keras.models import Model
import numpy as np
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255
x_train = np.reshape(x_train,(-1,28*28))
x_test = np.reshape(x_test,(-1,28*28))

encoded_img = 2
input = Input(shape=(784,))
x = Dense(128,activation='relu')(input)
x = Dense(64,activation='relu')(x)
x = Dense(10,activation='relu')(x)
encoded = Dense(encoded_img)(x)

decoded = Dense(10,activation='relu')(encoded)
decoded = Dense(64,activation='relu')(decoded)
decoded = Dense(128,activation='relu')(decoded)
decoded = Dense(784,activation='sigmoid')(decoded)

autoencoder = Model(inputs=input,outputs=decoded)
encoder = Model(inputs=input,outputs=encoded)
autoencoder.compile(optimizer="adam",loss='binary_crossentropy')
autoencoder.fit(x_train,x_train,epochs=30,batch_size=256,shuffle=True)
encoded_imgs = encoder.predict(x_test)
decoder_imgs = autoencoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, s=3)
plt.colorbar()
plt.show()

from keras.utils import plot_model
plot_model(autoencoder,show_shapes=True,to_file='autoencoder.png')
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
 
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoder_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
