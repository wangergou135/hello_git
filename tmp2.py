from keras.layers import Conv2D, MaxPooling2D, Input
import keras

input_img = Input(shape=(256, 256, 3))

tower_1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(tower_1)
print(tower_1.get_shape())

tower_2 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_img)
tower_2 = Conv2D(64, (5, 5), padding='same', activation='relu')(tower_2)
print(tower_2.get_shape())

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(64, (1, 1), padding='same', activation='selu')(tower_3)
print(tower_3.get_shape())

output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
print(output.get_shape())