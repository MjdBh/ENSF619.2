# This is just an example. Feel free to replace it with your own solution
from data_generator import DataGeneratorUnet
import glob
import matplotlib.pylab as plt

imgs_list_train = glob.glob("./Sample/Images/*.npy")
masks_list_train = glob.glob("./Sample/Masks/*.npy")

batch_size = 4

gen_train = DataGeneratorUnet(imgs_list_train, masks_list_train, batch_size=batch_size)
X, Y = gen_train.__getitem__(0)
plt.figure(figsize=(12, 18))
for ii in range(batch_size):
    plt.subplot(1, 4, ii + 1)
    plt.imshow(X[ii, :, :, 0], cmap="gray")
    plt.axis("off")
    plt.imshow(Y[ii, :, :, 0], cmap='jet', alpha=0.5, interpolation='none')
    plt.axis("off")
plt.show()
