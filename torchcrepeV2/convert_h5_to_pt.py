"""
Script to convert keras crepe model to torch
"""
import os
from tensorflow.keras.layers import Input, Reshape, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Dropout, Permute, Flatten, Dense
from tensorflow.keras.models import Model
import torch


capacity_multiplier = 32    # for "full"
layers = [1, 2, 3, 4, 5, 6]
filters = [n * capacity_multiplier for n in [32, 4, 4, 4, 8, 16]]
widths = [512, 64, 64, 64, 64, 64]
strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

x = Input(shape=(1024,), name='input', dtype='float32')
y = Reshape(target_shape=(1024, 1, 1), name='input-reshape')(x)

for l, f, w, s in zip(layers, filters, widths, strides):
    y = Conv2D(f, (w, 1), strides=s, padding='same',
                activation='relu', name="conv%d" % l)(y)
    y = BatchNormalization(name="conv%d-BN" % l)(y)
    y = MaxPool2D(pool_size=(2, 1), strides=None, padding='valid',
                    name="conv%d-maxpool" % l)(y)
    y = Dropout(0.25, name="conv%d-dropout" % l)(y)

y = Permute((2, 1, 3), name="transpose")(y)
y = Flatten(name="flatten")(y)
y = Dense(360, activation='sigmoid', name="classifier")(y)

model = Model(inputs=x, outputs=y)

package_dir = os.path.dirname(os.path.realpath(__file__))
filename = "assets/model-full.h5"
model.load_weights(os.path.join(package_dir, filename))

state_dict = {}

for i in range(1, 7):
    print("conv_list.{}.weight".format(i - 1), torch.tensor(model.get_layer("conv{}".format(i)).kernel.numpy()).permute(3, 2, 0, 1).shape)
    state_dict["conv_list.{}.weight".format(i - 1)] = torch.tensor(model.get_layer("conv{}".format(i)).kernel.numpy()).permute(3, 2, 0, 1)
    print("conv_list.{}.bias".format(i - 1), torch.tensor(model.get_layer("conv{}".format(i)).bias.numpy()).shape)
    state_dict["conv_list.{}.bias".format(i - 1)] = torch.tensor(model.get_layer("conv{}".format(i)).bias.numpy())
    print("batchnorm_list.{}.weight".format(i - 1), torch.tensor(model.get_layer("conv{}-BN".format(i)).gamma.numpy()).shape)
    state_dict["batchnorm_list.{}.weight".format(i - 1)] = torch.tensor(model.get_layer("conv{}-BN".format(i)).gamma.numpy())
    print("batchnorm_list.{}.bias".format(i - 1), torch.tensor(model.get_layer("conv{}-BN".format(i)).beta.numpy()).shape)
    state_dict["batchnorm_list.{}.bias".format(i - 1)] = torch.tensor(model.get_layer("conv{}-BN".format(i)).beta.numpy())
    print("batchnorm_list.{}.running_mean".format(i - 1), torch.tensor(model.get_layer("conv{}-BN".format(i)).moving_mean.numpy()).shape)
    state_dict["batchnorm_list.{}.running_mean".format(i - 1)] = torch.tensor(model.get_layer("conv{}-BN".format(i)).moving_mean.numpy())
    print("batchnorm_list.{}.running_var".format(i - 1), torch.tensor(model.get_layer("conv{}-BN".format(i)).moving_variance.numpy()).shape)
    state_dict["batchnorm_list.{}.running_var".format(i - 1)] = torch.tensor(model.get_layer("conv{}-BN".format(i)).moving_variance.numpy())
    print("===============")

state_dict["classifier.weight"] = torch.tensor(model.get_layer("classifier").kernel.numpy()).permute(1, 0)
state_dict["classifier.bias"] = torch.tensor(model.get_layer("classifier").bias.numpy())


torch.save(state_dict, 'assets/model-full-crepe.pt')

    


