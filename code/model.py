import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_attention import (
    SpectralNorm,
    SelfAttention
)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config


class ApplyNoise(nn.Module):
    """
        主要就是将给定的 `noise`, 缩放, 加入到我们的 image 中
        注意，这里的 noise 是同 channels 共享的
        但是我们的 scale 是针对 channel(feature map)进行缩放
    """
    def __init__(self, channels):
        super(ApplyNoise, self).__init__()
        self.scale = nn.Parameter(torch.zeros(channels))    # 先是从没有噪声，然后慢慢调整加入的

    def forward(self, x, noise=None):
        shape = x.shape
        if noise is None:
            noise = torch.randn(shape[0], 1, shape[2], shape[3])
        return x + noise.to(x.device) * self.scale.reshape(1, -1, 1, 1)


class FC(nn.Module):
    """
        从原理上来讲，他就是一个 Linear + ActFunc，但是在 Style 应用上，他又加入了
            ELR 和 Kaiming He Initialize
            两者都是为了防止我们训练过程中梯度出现问题，尤其是网络很深的时候
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            gain=2**0.5,        # Kaiming He Initialize 的分子 std = (0.5 ** 2) / ()
            use_wscale=False,   # 是否采用 ELR (equalized learning rate)
            lrmul=1.0,          # 有点抽象, learning rate multiplier for the mapping layers
                                # 主要就是作为一个乘法因子，来加快，或者是降低梯度更新速度
            bias=True
    ):
        super(FC, self).__init__()
        he_std = gain * (in_channels ** -0.5)   # Kaiming He Initializer
        if use_wscale:  # 使用 ELR
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        # 进行初始化
        # 这里需要注意，先进行 out_channels，然后是 in_channels, 先后权重维度的顺序和具体 pytorch api 有关
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, weight=self.weight * self.w_lrmul, bias=self.bias * self.b_lrmul)
        else:
            out = F.linear(x, weight=self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out


class ApplyStyle(nn.Module):
    """
        对应 论文 network 中 A 的部分，将我们的 W 转换成 Style_s, Style_b，然后应用到我们的 Image 上
        从实现的角度上来看，直接使用 Linear (FC) 即可
    """
    def __init__(
            self,
            latent_size,    # 输入 W 的大小
            channels,       # 合并到图片上的 channels，与我们转换成的 style 相关
            use_wscale      # ELR 是否在 FC 中使用
    ):
        super(ApplyStyle, self).__init__()
        self.affine_trans = FC(
            in_channels=latent_size,
            out_channels=channels * 2,
            gain=1.0,     # 这个 gain=1.0 给的很奇怪，不过尊重源码吧
            use_wscale=use_wscale
        )

    def forward(self, x, latent_w):
        style = self.affine_trans(latent_w)
        # 一定要助力这个 shape 的样子，因为后面的广播机制要求他这个样子
        # 后面这两个 1, 1 保证了不会报错，而且按照论文的公式走
        style = style.reshape(latent_w.shape[0], 2, -1, 1, 1)
        # 不知道这个 + 1.0，会有什么奇效可能是考虑到 FC weight 和 bias 的初始化，但是我们 FC 之后的 style_s 应该为1， style_b 为0
        x = x * (style[:, 0] + 1.0) + style[:, 1]
        return x


class Conv2d(nn.Module):
    """
        FC 是使用 Kaiming He Initialize 和 ELR + LeakyReLU 之后的 Lineaer
        Conv2d 也是同样使用 He Initialize 和 ELR，
        他具体是实现方法也是和 FC 差不多
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            gain=2 ** 0.5,      # Kaiming He initialize
            use_wscale=False,   # ELR
            lrmul=1.0,          # ELR
            bias=True,
            stride=1,
            padding=None
    ):
        super(Conv2d, self).__init__()
        self.stride = stride
        he_std = gain * (in_channels * kernel_size * kernel_size) ** (-0.5)     # Kaiming He standard deviation
        self.kernel_size = kernel_size

        self.padding = (kernel_size // 2 if padding is None else padding)
        # weight 权重的处理
        if use_wscale:      # enable ELR
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:               # not use ELR
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * init_std
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_channels)
            )
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv2d(
                x, self.weight * self.w_lrmul, self.bias * self.b_lrmul,
                stride=self.stride, padding=self.padding
            )
        else:
            return F.conv2d(
                x, self.weight * self.w_lrmul, stride=self.stride, padding=self.padding
            )


class Upscale2d(nn.Module):
    """
        很简单的一个上采用，我看 github 上使用的是手写的最近邻
        我看不如直接套用 torch.nn.functional 中的 bi-linear
    """
    def __init__(self, factor=2, gain=1):
        super(Upscale2d, self).__init__()
        assert factor > 1
        self.factor = factor
        self.gain = gain

    def forward(self, x):
        if self.gain != 1:
            x = self.gain * x
        # 直接调用 torch.nn.functional 中的库函数实现 bi-linear
        return F.interpolate(x, scale_factor=self.factor, mode='bilinear')


class PixelNorm(nn.Module):
    """
        在 stylegan 中,初始特征向量`z`会先经过`pixelnorm`再流向`mapping`层转换成线性无关的中间特征向量.
        就是对 channel 进行 归一化，仅仅只是对`std` 进行了处理，中期答辩写错了卧槽
    """
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # 注意是 dim = 1, 对 feature 进行操作
        # 乘以 * rsqrt 比 除以 / sqrt 好一些
        return x * torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + self.epsilon)


class InstanceNorm(nn.Module):
    """
        Instance Normalize, 主要是针对 H,W 这一部分进行归一化，既包括了 mean 的调整，还有 std 的部分
        具体的实现方法和上面的 PixelNorm 类似
    """
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, dim=(2, 3), keepdim=True)     # (batch_size, c, 1, 1) 然后进行广播，去除了均值项
        # 乘以 * rsqrt 比 除以 / sqrt 好一些
        return x * torch.rsqrt(torch.mean(x * x, dim=(2, 3), keepdim=True) + self.epsilon)


class LayerEpilogue(nn.Module):
    """
        Epilogue 在英文中，有后记的意思。 该层次就是 AdaIn 层次, 它包含的部分如下所示
            1. Noise 是否进行添加 (后面跟着激活函数)
            2. Style 的风格转换
            3. PixelNorm 和 Instance Norm 的使用
    """
    def __init__(
            self,
            channels,       # 图片的 channel 大小，noise style 层需要
            dlatent_size,   # w 的维度大小， style transformation 需要
            use_wscale,     # 是否使用 ELR
            use_noise,      # 这部分应该默认为 True
            use_pixel_norm,
            use_instance_norm,
            use_styles=True
    ):
        super(LayerEpilogue, self).__init__()
        if use_noise:
            self.apply_noise = ApplyNoise(channels=channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):
        """
            x 指的是从 const parameter 一路过来的， noise 是用于添加局部噪声的，
            dlatents_in_slice 是用于我们 style 转换的
        """
        x = self.apply_noise(x, noise)
        x = self.act(x)

        if self.pixel_norm is not None:
            x = self.pixel_norm(x)

        if self.instance_norm is not None:
            x = self.instance_norm(x)

        if self.style_mod is not None:
            assert dlatents_in_slice is not None
            x = self.style_mod(x, dlatents_in_slice)

        return x


class GBlock(nn.Module):
    """
        是 8X8, 16X16 或者是 32X32 这样的一块
        主要是包括上采样, `noise`， `style`等等部分，需要两个 `AdaIN`
        我们可以使用 `res` 进行计算 `channel` 特征图数量，也可以直接传递数值，就想之前复现的 `StyleGAN` 一样
    """
    def __init__(
            self,
            res,                # 主要就是定位作用，说明你在第几层
                                # 对`1024`人脸数据集而言，就是[3, 4, 5, ..10]， 4X4初始层单拿出来了，所以没有2
            use_wscale,         # ELR
            use_noise,          # 是否使用 noise 噪声
            use_pixel_norm,
            use_instance_norm,
            noise_input,        # 输入的特征，这是很多层的noise，一次输入这么多是方便我们后期的扩充
            dlatent_size=512,   # w 的大小， G_mapping 传过来的数值
            use_styles=True,    # affine style 部分
            f=None,             #
            factor=2,           # 进行 upsample 的 factor 因子
            fmap_base=8192,     # 与剩下两遍变量，一同计算channel的大小
            fmap_decay=1.0,
            fmap_max=512,
            has_sa=False,       # 是否含有self-attention
    ):
        super(GBlock, self).__init__()
        # 用于计算某一个层特征图，通道数大小的 lambda 函数, num_feature_maps
        self.nf = lambda stage: min(int(fmap_base / (2 ** (stage * fmap_decay))), fmap_max)

        self.res = res

        ##############################################################
        # self.noise_input = noise_input
        self.noise_input = [None for i in range(20)]

        # ============================================================== #
        #             这个是可以进行修改的点，毕竟我们的不再是1024维度的了
        # 而且这个 self.nf 一旦使用，就需要一直使用，否则可能会造成前后对接不上的效果
        # 不过我当让还是倾向于第一个选择，bi-linear 上采样 + conv 卷积，因为 transposed 可能会导致棋盘效应
        if res < 7:
            # 这里我直接给他使用的是 bi-linear + conv2d
            self.up_sample = nn.Sequential(
                Upscale2d(factor),
                Conv2d(
                    in_channels=self.nf(self.res - 3),      # res 第一次是3,2的时候是在`const input`那一层
                    out_channels=self.nf(self.res - 2),     # 通道数是必然降低了
                    kernel_size=3,      # 因为 stride = 1, 这里需要 kernel_size = 3 来保证图片像素大小的不变
                    stride=1,
                    # padding=1, Padding 参数不需要自己给，他已经给了
                )
            )
        else:
            self.up_sample = nn.ConvTranspose2d(
                in_channels=self.nf(self.res - 3),
                out_channels=self.nf(self.res - 2),
                kernel_size=4,
                stride=2,
                padding=1
            )

        self.adaIn1 = LayerEpilogue(
            channels=self.nf(self.res - 2), dlatent_size=dlatent_size, use_wscale=use_wscale,
            use_noise=use_noise, use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm,
            use_styles=use_styles
        )
        self.adaIn2 = LayerEpilogue(
            channels=self.nf(self.res - 2), dlatent_size=dlatent_size, use_wscale=use_wscale,
            use_noise=use_noise, use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm,
            use_styles=use_styles
        )
        self.conv1 = Conv2d(
            in_channels=self.nf(self.res - 2), out_channels=self.nf(self.res - 2), kernel_size=3
        )

        ##################################################
        ############# 加入 self-attenion 的部分 ############
        self.sa = SelfAttention(in_dim=self.nf(self.res - 2)) if has_sa else None

    def forward(self, x, dlatent):
        # res -> [3, 4, ..., 10]
        # 2 的话，被第一个 constant 的 block 给使用了
        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res*2-4], dlatent[:, self.res*2-4])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res*2-3], dlatent[:, self.res*2-3])

        if self.sa is not None:
            x = self.sa(x)
        return x


class DBlock(nn.Module):
    """
        主要是用于 Discriminator 中的 DBlock 部分
    """
    def __init__(self, in_channels, out_channels, has_sa=False):
        super(DBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3)
        self.leaky = nn.LeakyReLU(0.2)

        # #################################################
        # ============== 加入 self attention ==============
        self.sa = SelfAttention(in_dim=out_channels) if has_sa else None

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))

        if self.sa is not None:
            x = self.sa(x)
        return x


class G_mapping(nn.Module):
    """
        用于解耦的 `G_mapping` 网络，将我们原本关联较大的 `noise z` 经过 8 层 `MLP`，
        转换成了表达能力更强的 `W` 空间，而且他施加了一个技巧，预先使用了 `PixelNorm`
        在流向 `MLP` 之前，将 `Z` 转换成线性无关的中间特征向量
    """
    def __init__(
            self,
            mapping_fmaps=512,  # Z dimension
            dlatent_size=512,   # 隐空间大小，`W`
            resolution=1024,    # 这个 Resolution 用处不大
            normalize_latents=True,
            use_wscale=True,    # 默认使用了 wscale ELR 方法
            lrmul=0.01,         # 这个在论文中有提及， 0.01
            gain=2**(0.5)
    ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, use_wscale, lrmul),  # 1
            FC(dlatent_size, dlatent_size, gain, use_wscale, lrmul),        # 2
            FC(dlatent_size, dlatent_size, gain, use_wscale, lrmul),        # 3
            FC(dlatent_size, dlatent_size, gain, use_wscale, lrmul),        # 4
            FC(dlatent_size, dlatent_size, gain, use_wscale, lrmul),        # 5
            FC(dlatent_size, dlatent_size, gain, use_wscale, lrmul),        # 6
            FC(dlatent_size, dlatent_size, gain, use_wscale, lrmul),        # 7
            FC(dlatent_size, dlatent_size, gain, use_wscale, lrmul),        # 8
        )

        self.normalize_latents = normalize_latents
        self.pixel_norm = PixelNorm()

        # 1024 就表示一共有18层
        # 这个应该有扩展的，比如说我们的 医学图像数据更强`Layer` 显然需要一个 5 X 5 作为基准，然后增长到 320
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2

    def forward(self, x):
        """
            `x` 是一个 `noise input z`, 不过返回值不仅仅是其对应的 `w`，还包括了 `num_layers`
        """
        if self.normalize_latents:
            x = self.pixel_norm(x)
        return self.func(x)


class StyleGenerator(nn.Module):
    """
        `StyleGAN`的生成器部分，
        1. 完成 Progressive 的特性
        2. truncation，截断
        3. Style Mixing
    """
    def __init__(
            self,
            mapping_fmaps=512,
            dlatent_size=512,
            resolution=1024,
            fmap_base=8192,
            fmap_max=512,
            fmap_decay=1.0,
            num_channels=3,
            use_wscale=True,            # Enable equalized learning rate?
            use_pixel_norm=False,        # 对于 Synthesis Network 是否使用 pixel norm, g_mapping 必须使用 pixel norm
            use_instance_norm=True,     # 对于 Synthesis Network 是否使用 instance norm
            use_noise=True,             # Enable noise inputs?
            use_style=True              # Enable style inputs?
    ):
        super(StyleGenerator, self).__init__()

        # 这里我们的 stage 从 1 开始吧
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        # 下面这个部分使用的时候，需要修改
        # =========================================================== #
        # =========================================================== #
        self.resolution_log2 = int(np.log2(resolution))
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.
        self.num_layers = self.resolution_log2 * 2 - 2

        self.num_layers = int(np.log2(resolution // config.BASE_CONSTANT_IMAGE_SIZE)) * 2 + 2

        # 制造我们的 noise，但是这样的 noise 不就固定死了吗？？
        # 这种写法很是奇怪
        self.noise_inputs = []
        for layer_idx in range(self.num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            # self.noise_inputs.append(torch.randn(*shape).to("cuda"))

        # ========================================================== #
        # =====================    网络框架的定义    ================= #
        self.map = G_mapping(
            mapping_fmaps=mapping_fmaps, dlatent_size=dlatent_size, resolution=resolution,
            normalize_latents=True, use_wscale=use_wscale, lrmul=0.001, gain=2 ** 0.5
        )

        # 初始层次，第一层的定义
        self.starting_constant = nn.Parameter(
            torch.ones(1, self.nf(0), config.BASE_CONSTANT_IMAGE_SIZE, config.BASE_CONSTANT_IMAGE_SIZE)
        )
        self.initial_adain1 = LayerEpilogue(
            channels=self.nf(0), dlatent_size=dlatent_size, use_wscale=use_wscale,
            use_noise=use_noise, use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm,
            use_styles=use_style
        )
        self.initial_adain2 = LayerEpilogue(
            channels=self.nf(0), dlatent_size=dlatent_size, use_wscale=use_wscale,
            use_noise=use_noise, use_pixel_norm=use_pixel_norm, use_instance_norm=use_instance_norm,
            use_styles=use_style
        )
        self.init_conv = Conv2d(
            in_channels=self.nf(0),
            out_channels=self.nf(0),
            kernel_size=3,
        )
        self.init_rgb = Conv2d(
            in_channels=self.nf(0),
            out_channels=num_channels,      # 转到了 3 RGB 图片上
            kernel_size=1                   # 使用的是 1X1卷积
        )

        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.init_rgb]),
        )

        # 从 1 到 num_layers // 2 - 1 的遍历
        for cur_stage in range(1, self.num_layers // 2):
            has_sa = False
            for __from_end_count__ in config.HAS_SA:
                if cur_stage == self.num_layers // 2 - __from_end_count__:
                    has_sa = True
                    print('GTrue')
            t_g_block = GBlock(
                res=cur_stage + 2,          # 这个有点绕，主要是遗留的问题，其实可以统一的，但是我懒得改了，
                                            # 原本的 github 这地方有 bug，但是不影响他的使用
                use_wscale=use_wscale,
                use_noise=use_noise,
                use_pixel_norm=use_pixel_norm,
                use_instance_norm=use_instance_norm,
                noise_input=self.noise_inputs,    # 这个地方有点奇怪，我建议是舍弃使用这一部分，
                dlatent_size=dlatent_size,
                use_styles=use_style,
                fmap_base=fmap_base,
                fmap_decay=fmap_decay,
                fmap_max=fmap_max,
                has_sa=has_sa  # 最后两层才有 self-attention
            )
            t_rgb = Conv2d(
                in_channels=self.nf(cur_stage),
                out_channels=num_channels,  # 转到了 3 RGB 图片上
                kernel_size=1  # 使用的是 1X1卷积
            )
            self.prog_blocks.append(t_g_block)
            self.rgb_layers.append(t_rgb)

    def fade_in(self, alpha, upscaled, generated):
        # alpha 应该从0开始，增长到1，体现的是一个 progressive 过程
        assert 1.0 >= alpha >= 0.0 and upscaled.shape == generated.shape
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, z, alpha, steps):
        #############################################################
        ################ 可以考虑传递多个z来 style mixing ##############
        w = self.map(z)
        w = w.unsqueeze(1)
        w = w.repeat(1, self.num_layers, 1)

        # dlatents_in_slice 记得帮他选好
        x = self.initial_adain1(self.starting_constant, noise=None, dlatents_in_slice=w[:, 0])
        x = self.init_conv(x)
        x = self.initial_adain2(x, noise=None, dlatents_in_slice=w[:, 1])
        # out = self.initial_adain2(x, noise=None, dlatents_in_slice=w)

        #############################################################
        if steps == 0:
            return self.init_rgb(x)     # 注意这个是 `x`, 不是`out`
        for step in range(steps):
            pre = x
            x = self.prog_blocks[step](x, w)

        # 需要注意的是，他经过了RGB这个东西
        final_upscaled = self.rgb_layers[steps-1](F.interpolate(pre, scale_factor=2, mode='bilinear'))
        final_out = self.rgb_layers[steps](x)       # 妈的, 这个 steps 别写成 step 啊！！！！！
        return self.fade_in(alpha, final_upscaled, final_out)


class StyleDiscriminator(nn.Module):
    def __init__(
            self,
            resolution=1024,        # 这个地方后期，修改数据集需要修改的
            fmap_base=8192,
            fmap_max=512,
            fmap_decay=1.0,
            num_channels=3,
            f=None
    ):
        super(StyleDiscriminator, self).__init__()
        #
        self.nf = lambda stage: min(int(fmap_base / (2 ** (fmap_decay * stage))), fmap_max)

        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        # assert resolution == 2 ** self.resolution_log2 and resolution >= 4

        self.num_layers = int(np.log2(resolution / config.BASE_CONSTANT_IMAGE_SIZE)) * 2 + 2

        #########################################################################
        # ============================= 定义网络 =============================== #
        # 该部分和我们 Generator 中网络的定义很像，反过来就可以了
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        # stage 并不会取到 0, 而且是从 num_layers - 1 开始
        for stage in range(self.num_layers // 2 - 1, 0, -1):   # 18 -> 9(from 4...1024), 14 -> 7(from 5...320)
            conv_in = self.nf(stage)    #
            conv_out = self.nf(stage-1)
            has_sa = False
            for __from_end_count__ in config.HAS_SA:
                if stage == self.num_layers // 2 - __from_end_count__:
                    has_sa = True
                    print('DTrue')

            self.prog_blocks.append(
                DBlock(
                    conv_in, conv_out,
                    has_sa=has_sa
                )
            )
            self.rgb_layers.append(
                Conv2d(num_channels, conv_in, kernel_size=1, stride=1)
            )

        self.initial_rgb = Conv2d(
            num_channels, self.nf(0), kernel_size=1, stride=1
        )
        self.rgb_layers.append(self.initial_rgb)

        # 这个是用于下采样的 layer
        self.avg_pool = nn.AvgPool2d(
            kernel_size=2, stride=2
        )

        # 这是用于 4 X 4的 block部分，他和其他层次不一样的点在于，它使用了 minibatch-std,
        # 也就是说，在第一层的时候， in_channels + 1 是因为加入了 minibatch-std 这一个操作
        # 该操作主要是降低模式崩溃的风险
        in_channels = self.nf(0)
        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),         # 4X4
            nn.LeakyReLU(0.2),
            Conv2d(
                in_channels,
                in_channels,
                kernel_size=config.BASE_CONSTANT_IMAGE_SIZE,    # 记得他的作用是放缩为 1X1,所以说这里的起始图片大小改一下
                padding=0, stride=1
            ),   # 1X1
            nn.LeakyReLU(0.2),
            Conv2d(
                in_channels, 1, kernel_size=1, padding=0, stride=1
            ),  # 使用 Conv 而不是 Linear 主要是为了降低复杂度
        )

    def fade_in(self, alpha, downscaled, out):
        """
            Used to fade in downscaled using avg pooling and output from CNN
            而且因为他是从 0 开始的，所以说需要注意的是， downscaled 乘以的是 (1-alpha)，这才满足我们的 fade in 操作
        """
        # alpha should be scalar within [0, 1], and downscaled.shape == out.shape
        assert 0.0 <= alpha <= 1.0 and downscaled.shape == out.shape
        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x):
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        # we take the std for each example (across all channels, and pixels) then we repeat it
        # for a single channel and concatenate it with the image. In this way the discriminator
        # will get information about the variation in the batch/image
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x, alpha, steps):
        """
            Steps 指的是正这走的 steps， 0 表示 4X4阶段, 1表示 8X8阶段
            pro_blocks 先放置的是 1024， 再放置的是 512 ...，但是他并没有放置 4X4的部分
        """
        # steps = 0, cur_step = len(self.rgb_layers) - 1 = len(self.prog_blocks)
        # steps = 1, cur_step = len(self.rgb_layers) - 2 = len(self.prog_blocks) - 1
        # 从而可以找到下面这个公式
        cur_step = len(self.prog_blocks) - steps

        # convert from rgb as initial step, this will depend on
        # the image size (each will have it's on rgb layer)
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:  # i.e, image is 4x4
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        # because prog_blocks might change the channels, for down scale we use rgb_layer
        # from previous/smaller size which in our case correlates to +1 in the indexing
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))    # 这个下采用是为了 fade-in 操作
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        # the fade_in is done first between the downscaled and the input
        # this is opposite from the generator
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


if __name__ == '__main__':
    test_apply_noise = False
    test_FC = False
    test_apply_style = False
    test_conv2d = False
    test_pixel_norm = False
    test_instance_norm = False
    test_layer_epilogue = False
    test_GBlock = False
    test_G_mapping = False

    test_generator = False
    test_discriminator = True

    if test_apply_noise:
        applyNoise = ApplyNoise(3)
        x = torch.randn(4, 3, 320, 320)
        applyNoise(x)
        print(x.shape)

    if test_FC:
        x = torch.randn(64, 100)
        linear = FC(100, 10, use_wscale=True)
        y = linear(x)
        print(y.shape)

    if test_apply_style:
        applyStyle = ApplyStyle(512, 64, True)
        w = torch.randn(16, 512)
        img = torch.randn(16, 64, 40, 40)
        out = applyStyle(img, w)
        print(out.shape)

    if test_conv2d:
        x = torch.randn(8, 3, 80, 80)
        conv2d = Conv2d(3, 10, kernel_size=5, stride=2)
        print(conv2d(x).shape)

    if test_pixel_norm:
        img = torch.ones(54, 10, 320, 320) * 2
        pixel_norm = PixelNorm()
        img2 = pixel_norm(img)
        # print(img)
        # print(img2)
        print(img.shape)
        print(img2.shape)

    if test_instance_norm:
        img = torch.ones(54, 10, 320, 320) * 2
        instance_norm = InstanceNorm()
        img2 = instance_norm(img)

        print(img.shape)
        print(img2.shape)

    if test_layer_epilogue:
        layer_epilogue = LayerEpilogue(100, 512, True, True, True, True, True)
        img = torch.randn(16, 100, 80, 80)
        w = torch.randn(16, 512)
        # noise = torch.randn(16, 1, 64, 64)
        y = layer_epilogue(img, None, w)       # 注意，参数的时候，先是 noise 再是 w
        print(y.shape)

    if test_GBlock:
        noise_input = [
            None for i in range(20)
        ]
        gblock1 = GBlock(
            3, True, True, True, True, noise_input=noise_input,
            dlatent_size=512, use_styles=True
        )
        gblock2 = GBlock(
            4, True, True, True, True, noise_input=noise_input,     # 注意这个 res 别写错了
            dlatent_size=512, use_styles=True
        )
        img = torch.randn((4, 512, 4, 4))
        dlatent = torch.randn(4, 18, 512)   # batch_size, 18/14 layers, dlatent
        img1 = gblock1(img, dlatent)
        print('the first success')
        print(img1.shape)

        img2 = gblock2(img1, dlatent)
        print(img2.shape)

    if test_G_mapping:
        g_mapping = G_mapping()
        z = torch.randn((4, 512))
        w = g_mapping(z)
        print(w.shape)

    if test_generator:
        generator = StyleGenerator(
            resolution=320
        )
        z = torch.randn((4, 512))
        alpha = 0.5
        steps = 6
        img = generator(z, alpha, steps)
        print(img.shape)
        print('this is a debug point')

    if test_discriminator:
        # config.DEVICE = torch.device('cpu')
        generator = StyleGenerator(
            resolution=config.RESOLUTION,
            fmap_base=config.FMAP_BASE,
            fmap_decay=config.FMAP_DECAY,
            fmap_max=config.FMAP_MAX,
            num_channels=config.NUM_CHANNELS,
        ).to(config.DEVICE)
        discriminator = StyleDiscriminator(
            resolution=config.RESOLUTION,
            fmap_base=config.FMAP_BASE,
            fmap_max=config.FMAP_MAX,
            fmap_decay=config.FMAP_DECAY,
            num_channels=config.NUM_CHANNELS
        ).to(config.DEVICE)
        z = torch.randn((1, 512)).to(config.DEVICE)
        alpha = 0.5

        with torch.cuda.amp.autocast():
            for steps in range(5, 6):  # 0:4, 1:8, 2:16, 3:32, 4:64, ... 6: 256, 8:1024
                # 0:5, 1:10, 2:20, 3:40, 4:80, 5:160, 6:320
                print(f'==>({steps}):')
                img = generator(z, alpha, steps)
                print('\timg:', img.shape)
                judge = discriminator(img, alpha, steps)
                print('\tjud:', judge.shape)
