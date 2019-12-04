import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class EncDec(nn.Module):

    def __init__(self):
        super(EncDec, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(64, 32, 3, stride=1),  # b, 16, 5, 5
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(32, 16, 3, stride=1, padding=1),  # b, 8, 15, 15
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),  # b, 1, 28, 28
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Upsample(mode='bilinear', scale_factor=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        #return self.forward_(x)

    def forward_(self, x):
        num_epochs = 5000
        learning_rate = 1e-2
        model = EncDec()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            print(epoch)
            x = Variable(x)
            # ===================forward=====================
            output = self.encoder(x)
            output = self.decoder(output)
            loss = criterion(output, x)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            x = output
            # ===================log========================
            #print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data[0]))
            """if epoch % 10 == 0:
                pic = to_img(output.cpu().data)
                save_image(pic, './dc_img/image_{}.png'.format(epoch))"""
        return output
