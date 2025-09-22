import torch
import torch.nn as nn
from .guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from .guided_diffusion.respace import SpacedDiffusion, space_timesteps
from .guided_diffusion.resample import UniformSampler
import math


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x*torch.sigmoid(x)


class ConvBlockOneStage(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, dropout):
        super(ConvBlockOneStage, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(n_filters_in, n_filters_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(n_filters_out, n_filters_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters_out),
            nn.LeakyReLU()
        )        

    def forward(self, x):
        out = self.conv_conv(x)
        return out


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        self.ops = nn.ModuleList()
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            self.ops.append(ConvBlockOneStage(input_channel, n_filters_out, dropout=0.1))

    def forward(self, x):

        for layer in self.ops:
            x = layer(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.ReLU(inplace=True))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.ReLU(inplace=True))

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.ReLU(inplace=True))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose2d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            ops.append(nn.ReLU(inplace=True))

        self.ops = nn.Sequential(*ops)

    def forward(self, x):
        x = self.ops(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, normalization='none', has_dropout=False, dropout_rate=0.5):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv2d(n_filters, n_classes, 1, padding=0)

    def forward(self, x1, x2, x3, x4, x5):
        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine(x8_up)

        out = self.out_conv(x9)
        return out


class Encoder(nn.Module):
    def __init__(self, n_filters, normalization, has_dropout, dropout_rate):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.dropout = nn.Dropout2d(p=dropout_rate, inplace=True)

    def forward(self, input):
        x1 = input
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)

        return x1, x2, x3, x4, x5




class TembFusion(nn.Module):
    def __init__(self, n_filters_out):
        super(TembFusion, self).__init__()
        self.temb_proj = torch.nn.Linear(512, n_filters_out)
    def forward(self, x, temb):
        if temb is not None:
            x = x + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        else:
            x =x
        return x



class TimeStepEmbedding(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, dropout_p):
        super(TimeStepEmbedding, self).__init__()

        self.embed_dim_in = 128
        self.embed_dim_out = self.embed_dim_in * 4

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.embed_dim_in,
                            self.embed_dim_out),
            torch.nn.Linear(self.embed_dim_out,
                            self.embed_dim_out),
        ])

        self.emb = ConvBlockTemb(n_filters_in, n_filters_out, dropout_p)

    def forward(self, x, t=None, image=None):
        if t is not None:
            temb = get_timestep_embedding(t, self.embed_dim_in)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)

        else:
            temb = None


        if image is not None :
            x = torch.cat([image, x], dim=1)
        x = self.emb(x, temb)

        return x, temb


class ConvBlockTembOneStage(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, dropout_p, temb_flag=True):
        super(ConvBlockTembOneStage, self).__init__()

        self.temb_flag = temb_flag

        self.conv_conv = nn.Sequential(
            nn.Conv2d(n_filters_in, n_filters_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(n_filters_out, n_filters_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters_out),
            nn.LeakyReLU()
        )

        self.temb_fusion =  TembFusion(n_filters_out)


    def forward(self, x, temb):

        norm = self.conv_conv(x)
        if self.temb_flag:
            out = self.temb_fusion(norm, temb)
        else:
            out = norm
        return out


class ConvBlockTemb(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, dropout_p):
        super(ConvBlockTemb, self).__init__()

        self.ops = nn.ModuleList()
        self.ops.append(ConvBlockTembOneStage(n_filters_in, n_filters_out, dropout_p, temb_flag=False))

    def forward(self, x, temb):

        for layer in self.ops:
            x = layer(x, temb)
        return x


class DownsamplingConvBlockTemb(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, dropout_p):
        super(DownsamplingConvBlockTemb, self).__init__()

        self.ops = nn.ModuleList()
        self.ops.append(nn.Conv2d(n_filters_in, n_filters_out, kernel_size=2, stride=2))
        self.ops.append(nn.BatchNorm2d(n_filters_out))
        self.ops.append(nn.LeakyReLU())
        self.ops.append(nn.Dropout(dropout_p))
        self.ops.append(TembFusion(n_filters_out))

    def forward(self, x, temb):
        for layer in self.ops:
            if layer.__class__.__name__ == "TembFusion":
                x = layer(x, temb)
            else:
                x = layer(x)
        return x

class UpsamplingDeconvBlockTemb(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(UpsamplingDeconvBlockTemb, self).__init__()

        self.ups = nn.ModuleList()
        self.ups.append(nn.Conv2d(n_filters_in, n_filters_out, kernel_size=1))
        self.ups.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.ups.append(TembFusion(n_filters_out))
        self.convs = ConvBlockTemb(n_filters_out*2, n_filters_out, dropout_p=0.1)

    def forward(self, x, x1, temb):
        for layer in self.ups:
            if layer.__class__.__name__ == "TembFusion":
                x = layer(x, temb)
            else:
                x = layer(x)
        x = torch.cat([x, x1], dim=1)

        return self.convs(x, temb)


class Encoder_denoise(nn.Module):
    def __init__(self, n_filters, dropout_rate):
        super(Encoder_denoise, self).__init__()

        self.block_one_dw = DownsamplingConvBlockTemb(n_filters[0], n_filters[0], dropout_rate[0])

        self.block_two = ConvBlockTemb(n_filters[0], n_filters[1], dropout_rate[1])
        self.block_two_dw = DownsamplingConvBlockTemb(n_filters[1], n_filters[1], dropout_rate[1])

        self.block_three = ConvBlockTemb(n_filters[1], n_filters[2], dropout_rate[2])
        self.block_three_dw = DownsamplingConvBlockTemb(n_filters[2], n_filters[2], dropout_rate[2])

        self.block_four = ConvBlockTemb(n_filters[2], n_filters[3], dropout_rate[3])
        self.block_four_dw = DownsamplingConvBlockTemb(n_filters[3], n_filters[3], dropout_rate[3])

        self.block_five = ConvBlockTemb(n_filters[3], n_filters[4], dropout_rate[4])

    def forward(self, x1, temb):

        x1_dw = self.block_one_dw(x1, temb)

        x2 = self.block_two(x1_dw, temb)
        x2_dw = self.block_two_dw(x2, temb)

        x3 = self.block_three(x2_dw, temb)
        x3_dw = self.block_three_dw(x3, temb)

        x4 = self.block_four(x3_dw, temb)
        x4_dw = self.block_four_dw(x4, temb)

        x5 = self.block_five(x4_dw, temb)

        return x1, x2, x3, x4, x5, temb

class Decoder_denoise(nn.Module):
    def __init__(self, n_classes=2, n_filters=16, normalization='none', has_dropout=False, dropout_rate=0.5):
        super(Decoder_denoise, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlockTemb(n_filters[4], n_filters[3])
        self.block_six_up = UpsamplingDeconvBlockTemb(n_filters[3], n_filters[2])
        self.block_seven_up = UpsamplingDeconvBlockTemb(n_filters[2], n_filters[1])
        self.block_eight_up = UpsamplingDeconvBlockTemb(n_filters[1], n_filters[0])

        self.out_conv = nn.Conv2d(n_filters[0], n_classes, 1, padding=0)

    def forward(self, x1, x2, x3, x4, x5, temb):
        x5_up = self.block_five_up(x5, x4, temb)
        x6_up = self.block_six_up(x5_up, x3, temb)
        x7_up = self.block_seven_up(x6_up, x2, temb)
        x8_up = self.block_eight_up(x7_up, x1, temb)

        out = self.out_conv(x8_up)
        return out


class DenoiseModel(nn.Module):
    def __init__(self, n_classes, n_channels, n_filters, dropout_rate, superpixel_block):
        super().__init__()
        self.embedding_diffusion = TimeStepEmbedding(superpixel_block+n_channels, n_filters[0], dropout_p=0.5)
        self.encoder = Encoder_denoise(n_filters, dropout_rate)
        self.decoder = Decoder_denoise(n_classes, n_filters, dropout_rate)

    def forward(self, x: torch.Tensor, t, image=None):

        x, temb = self.embedding_diffusion(x, t=t, image=image)
        x1, x2, x3, x4, x5, temb = self.encoder(x, temb)
        out = self.decoder(x1, x2, x3, x4, x5, temb)
        return out


class DiffVNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=[16, 32, 64, 128, 256], dropout_rate=[0.05, 0.1, 0.2, 0.3, 0.5], superpixel_block=16):
        super(DiffVNet, self).__init__()
        self.n_classes = n_classes
        self.time_steps = 1000

        self.denoise_model = DenoiseModel(n_classes, n_channels, n_filters ,dropout_rate, superpixel_block)

        betas = get_named_beta_schedule("linear", self.time_steps, use_scale=True)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(self.time_steps, [self.time_steps]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sampler = UniformSampler(self.time_steps)


    def forward(self, image=None, x=None, pred_type=None, step=None):

        if pred_type == "q_sample":
            noise = torch.randn_like(x.float()).cuda()
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "D_xi_l":
            x, temb = self.denoise_model.embedding_diffusion(x, t=step, image=image)
            x1, x2, x3, x4, x5, temb = self.denoise_model.encoder(x, temb)
            return self.denoise_model.decoder(x1, x2, x3, x4, x5, temb)


if __name__ == "__main__":
    model = DiffVNet(
        n_channels=3,
        n_classes=9,
        superpixel_block=4
    ).cuda()

    label_1hot = torch.randn(2, 4, 64, 64).cuda()
    input_gpu = torch.randn(2, 3, 64, 64).cuda()
    x_t, t, noise = model(x=label_1hot, pred_type="q_sample")
    x0_pred = model(x=x_t,step=t,image=input_gpu,pred_type="D_xi_l")  
    print(x0_pred.shape) 