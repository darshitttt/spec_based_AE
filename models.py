import torch
import torch.nn as nn

class logMelAE(nn.Module):

    def __init__(self):
        super(logMelAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(),

            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
    
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(3), stride=1, padding=1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(32, 64, kernel_size=(3), stride=1, padding=1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(64, 128, kernel_size=(3), stride=1, padding=1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 256, kernel_size=(3), stride=1, padding=1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 512, kernel_size=(3), stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(512, 1, kernel_size=(3,2), stride=2, padding=1, output_padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
class BL_Decoder_Classifier(nn.Module):

    def __init__(self):
        super(BL_Decoder_Classifier, self).__init__()

        self.classifier= nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(858, 200),
            nn.ReLU(),
            nn.Linear(200, 10)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.classifier(x)
        y = self.sigmoid(x)

        return y
    

class logMel_classifier(nn.Module):

    def __init__(self):
        super(logMel_classifier, self).__init__()

        self.classifier = nn.Sequential(nn.Conv2d(1 ,32 ,kernel_size=2),
                     nn.ReLU(),
                     nn.MaxPool2d(2),
                     nn.Conv2d(32 ,64 ,kernel_size=2),
                     nn.ReLU(),
                     nn.MaxPool2d(2),
                     nn.Conv2d(64 ,128 ,kernel_size=2),
                     nn.ReLU(),
                     nn.Conv2d(128 ,32 ,kernel_size=2),
                     nn.ReLU(),
                     nn.Conv2d(32 ,1 ,kernel_size=2),
                     nn.ReLU(),
                     nn.Flatten(),
                     nn.Linear(354 ,220),
                     nn.ReLU(),
                     nn.Linear(220 ,10))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.classifier(x)
        y = self.sigmoid(x)

        return y