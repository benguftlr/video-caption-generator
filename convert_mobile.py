import torch
from Inception import inception_v3

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

encoder = inception_v3(pretrained=True)
encoderGRU=torch.load("model/14_epoch_encoder.pth")
decoder = torch.load("model/14_epoch_decoder.pth")


class Generator(nn.Module):

    def __init__(self, e,eGRU,d):
        super().__init__()
        self.encoder = e
        self.encoderGRU= eGRU
        self.decoder = d

    def forward(self,inputImage0, inputImage1,inputImage2,inputImage3,inputImage4,inputImage5,inputImage6,inputImage7):
        features=torch.zeros(8,2048)

        features[0] = self.encoder.forward(inputImage0)
        features[1] = self.encoder.forward(inputImage1)
        features[2] = self.encoder.forward(inputImage2)
        features[3] = self.encoder.forward(inputImage3)
        features[4] = self.encoder.forward(inputImage4)
        features[5] = self.encoder.forward(inputImage5)
        features[6] = self.encoder.forward(inputImage6)
        features[7] = self.encoder.forward(inputImage7)

        encoder_hidden = torch.zeros(1, 1, 256)
        for i in range(8):
            encoder_output,encoder_hidden = self.encoderGRU((features[i].unsqueeze(0)).unsqueeze(0),encoder_hidden)

        decoder_hidden = encoder_hidden
        inputtoken = torch.ones(1, 1).type(torch.LongTensor)
        output = torch.zeros(30)
        for i in range(30):
            out, decoder_hidden = self.decoder(inputtoken, decoder_hidden)
            out = out.argmax(dim=1)
            output[i] = out
            inputtoken = out.unsqueeze(0)
        return output


encoder.eval()
encoderGRU.eval()
decoder.eval()
decoder.cpu()
encoderGRU.cpu()

generator = Generator(e=encoder,eGRU=encoderGRU, d=decoder)
encoder_input = torch.zeros(1,3, 299, 299)
generator_input0 = torch.zeros(1,3, 299, 299)
generator_input1 = torch.zeros(1,3, 299, 299)
generator_input2 = torch.zeros(1,3, 299, 299)
generator_input3 = torch.zeros(1,3, 299, 299)
generator_input4 = torch.zeros(1,3, 299, 299)
generator_input5 = torch.zeros(1,3, 299, 299)
generator_input6 = torch.zeros(1,3, 299, 299)
generator_input7 = torch.zeros(1,3, 299, 299)


decoder_input1 = torch.tensor([[0]])
decoder_input2 = torch.zeros(1, 1, 256)

# dynamic quantization can be applied to the decoder for its nn.Linear parameters
quantized_decoder = torch.quantization.quantize_dynamic(decoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

traced_encoder = torch.jit.trace(encoder, encoder_input)
traced_generator = torch.jit.trace(generator, (generator_input0,generator_input1,generator_input2,generator_input3,generator_input4,generator_input5,generator_input6,generator_input7))
traced_decoder = torch.jit.trace(quantized_decoder, (decoder_input1, decoder_input2))

from torch.utils.mobile_optimizer import optimize_for_mobile

# traced_encoder_optimized = optimize_for_mobile(traced_encoder)
# traced_encoder_optimized.save("optimized_encoder_150k.pth")
#traced_encoder.save("encoder.pth")
traced_generator.save("generator.pth")

# traced_decoder_optimized = optimize_for_mobile(traced_decoder)
# traced_decoder_optimized.save("optimized_decoder_150k.pth")
#traced_decoder.save("decoder.pth")