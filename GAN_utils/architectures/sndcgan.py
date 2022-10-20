from torch import nn



class Generator(nn.Module):
    def __init__(self, num_chan, h, w, input_dim=128):

        self.init_h = h//8
        self.init_w = w//8
        self.h = h
        self.w = w

        super(Generator, self).__init__()
        self.preprocess = nn.Sequential(
            nn.Linear(input_dim, self.init_h*self.init_w*512, bias = False),
            nn.BatchNorm1d(self.init_h*self.init_w*512),
            nn.ReLU(True)
        )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, bias = False, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, bias = False, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, bias = False, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(64, num_chan, 3, stride=1, padding = 1)

        self.tanh = nn.Tanh()
        self.num_chan = num_chan

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 512, self.init_h, self.init_w)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, self.num_chan, self.h, self.w)




class Discriminator(nn.Module):
    def __init__(self, num_chan, h, w, DIM):
        super(Discriminator, self).__init__()
       
        self.DIM = DIM
        self.final_h = h//8
        self.final_w = w//8
        main = nn.Sequential(nn.Conv2d(num_chan, DIM, 3, 1,padding = 1),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(DIM, 2*DIM, 4, 2, padding = 1),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(2*DIM, 2*DIM, 3, 1, padding = 1),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(2 * DIM, 4 * DIM, 4, 2, padding = 1),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(4 * DIM, 4 * DIM, 3, 1, padding = 1),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(4 * DIM, 8 * DIM, 4, 2, padding = 1),
                nn.LeakyReLU(negative_slope = 0.1),
                nn.Conv2d(8 * DIM, 8 * DIM, 3, 1, padding = 1),
                nn.LeakyReLU(negative_slope = 0.1))
        self.main = main
        self.linear = nn.Linear(self.final_h*self.final_w*8*DIM, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, self.final_h*self.final_w*8*self.DIM)
        output = self.linear(output)
        return output