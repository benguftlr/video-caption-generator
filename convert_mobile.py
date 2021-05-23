import torch
from Inception import inception_v3

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

encoder = inception_v3(pretrained=True)
encoderGRU=torch.load("14_epoch_encoder.pt")
decoder = torch.load("14_epoch_decoder.pt")


class Generator(nn.Module):

    def __init__(self, e,e1,d):
        super().__init__()
        self.encoder = e
        self.encoder1= e1
        self.decoder = d

    def forward(self, inputImage1):
        features=torch.zeros(8,2048)
        for j,input_image in inputImage:
            features[j] = self.encoder.forward(input_image)

        encoder_hidden = torch.zeros(1, 1, 2048)
        for i in range(8):
            encoder_output,encoder_hidden = self.encoder1(features[i,:],encoder_hidden)

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
encoder1.eval()
decoder.eval()
decoder.cpu()
encoder1.cpu()

generator = Generator(e=encoder,e1=encoder1, d=decoder)
encoder_input = torch.zeros(1, 3, 80, 80)
generator_input = torch.zeros(1, 3, 80, 80)

decoder_input1 = torch.tensor([[0]])
decoder_input2 = torch.zeros(1, 1, 2048)

# dynamic quantization can be applied to the decoder for its nn.Linear parameters
quantized_decoder = torch.quantization.quantize_dynamic(decoder, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)

traced_encoder = torch.jit.trace(encoder, encoder_input)
traced_generator = torch.jit.trace(generator, generator_input)
traced_decoder = torch.jit.trace(quantized_decoder, (decoder_input1, decoder_input2))

from torch.utils.mobile_optimizer import optimize_for_mobile

# traced_encoder_optimized = optimize_for_mobile(traced_encoder)
# traced_encoder_optimized.save("optimized_encoder_150k.pth")
traced_encoder.save("encoder.pth")
traced_generator.save("generator.pth")

# traced_decoder_optimized = optimize_for_mobile(traced_decoder)
# traced_decoder_optimized.save("optimized_decoder_150k.pth")
traced_decoder.save("decoder.pth")