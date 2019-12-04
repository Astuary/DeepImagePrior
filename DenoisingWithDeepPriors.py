# This code is part of:
#
#   CMPSCI 670: Computer Vision
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from DeepImagePrior import DenseNet
from utils import imread
from torch.autograd import Variable
from skimage import color

# Load clean and noisy image
#im = imread('../data/denoising/saturn.png')
#noise1 = imread('../data/denoising/saturn-noisy.png')
im = imread('../data/denoising/lena.png')
noise1 = imread('../data/denoising/lena-noisy.png')
#im = color.rgb2gray(imread('../data/demosaic/puppy.jpg'))
#noise1 = color.rgb2gray(imread('../data/demosaic/pupp-baseline-dmsc.jpg'))
#im = imread('../data/demosaic/puppy.jpg')
#noise1 = imread('../data/demosaic/pupp-baseline-dmsc.jpg')

"""if im.shape[1] % 2 == 1:
     im = np.resize(im, (im.shape[0], im.shape[1]-1))
     noise1 = np.resize(noise1, (noise1.shape[0], noise1.shape[1]-1))

if im.shape[0] % 2 == 1:
     im = np.resize(im, (im.shape[0]-1, im.shape[1]))
     noise1 = np.resize(noise1, (noise1.shape[0]-1, noise1.shape[1]))"""

"""s = im.shape
im = np.resize(im, (384, 256, 3))
noise1 = np.resize(noise1, (384, 256, 3))"""

error1 = ((im - noise1)**2).sum()

print ('Noisy image SE: {:.2f}'.format(error1))

plt.figure(1)

plt.subplot(121)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(122)
plt.imshow(noise1, cmap='gray')
plt.title('Noisy image SE {:.2f}'.format(error1))

plt.show(block=False)


################################################################################
# Denoising algorithm (Deep Image Prior)
################################################################################

# Loads noisy image and sets it to the appropriate shape
noisy_img = torch.FloatTensor(noise1).unsqueeze(0).unsqueeze(0).transpose(2, 3)
clean_img = torch.FloatTensor(im).unsqueeze(0).unsqueeze(0).transpose(2,3)

#Create network
net = DenseNet(noisy_img.shape[2], noisy_img.shape[3])

# Creates \eta (noisy input)
eta = torch.randn(*noisy_img.size())
eta.detach()

num_epochs = 5
learning_rate = 1e-1
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
error = np.zeros((num_epochs, 1))

for epoch in range(num_epochs):
    print(epoch)
    eta = Variable(eta)
    # ===================forward=====================
    output = net(eta)
    loss = criterion(output, noisy_img)
    error[epoch] = loss.data
    # ===================backward====================
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #eta = output
    # ===================log========================
    #print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))

# Shows final result
out = net(eta)
out_img = out[0, 0, :, :].transpose(0,1).detach().numpy()

"""im = np.resize(im, s)
noise1 = np.resize(noise1, s)
out_img = np.resize(out_img, s)"""

error1 = ((im - noise1)**2).sum()
error2 = ((im - out_img)**2).sum()

plt.figure(3)
plt.axis('off')

plt.subplot(131)
plt.imshow(im, cmap='gray')
plt.title('Input')

plt.subplot(132)
plt.imshow(noise1, cmap='gray')
plt.title('SE {:.2f}'.format(error1))

plt.subplot(133)
plt.imshow(out_img, cmap='gray')
plt.title('SE {:.2f}'.format(error2))

plt.show()

#plt.savefig('')

"""plt.figure(4)
fig = plt.gcf()
plt.plot(np.arange(1, num_epochs+1), error, marker='o', linestyle='dashed', label='Training Error')
plt.legend(prop={'size': 22})
plt.xlabel('Number of Epochs/ Iterations', fontsize=22)
plt.ylabel('Training Squared Error', fontsize=22)
plt.title(r'Training Error as a function of the number of iterations', fontsize=22)
mng = plt.get_current_fig_manager()
mng.window.showMaximized()
plt.show()
fig.savefig('q2b_train_saturn.png')"""
