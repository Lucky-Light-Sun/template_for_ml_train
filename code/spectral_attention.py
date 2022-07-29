import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from spectral_attention import *
from torch.nn import Parameter
# import config


def l2normalize(t, eps=1e-12):
    """
    正则化，主要是让其 l2_ = 1
    :param t: 作用的 tensor 对象
    :param eps: 防止除以0操作的epsilon
    :return: 返回正则化之后的对象
    """
    return t / (t.norm() + eps)


class SpectralNorm(nn.Module):
    """
    频谱归一化，关键是在于如何计算出谱范数(利用 u, v 进行迭代的方法)
    """
    def __init__(self, module: nn.Module, name='weight', power_iterations=1):
        """
        self 参数设置，并且初始化 self.module 的属性，方便我们后期的操作
        """
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        nn.init.xavier_uniform_(self.module.weight.data)    # 这里初始化了
        if not self._made_u_v_w():
            self._make_u_v_w()

    def _make_u_v_w(self):
        w = getattr(self.module, self.name)

        w_hat = nn.Parameter(w.data).to(torch.float32)
        height = w_hat.shape[0]
        width = np.prod(w_hat.shape) // height      # 一定要注意，他是整除
        # 一定要注意这个设备的转化
        # u = l2normalize(torch.randn((height,), dtype=w_hat.data.dtype, device=w_hat.data.device))
        # v = l2normalize(torch.randn((width,), dtype=w_hat.data.dtype, device=w_hat.data.device))
        # 如果使用了 AMP autocast，这一部分很可能被转化
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False).to(torch.float32)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False).to(torch.float32)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)

        del self.module._parameters[self.name]  # 是一个 OrderedDict，将我们的 weight 删除

        # 注册一下这几个变量 weight_u, weight_v, weight_bar
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_w_hat", w_hat)

        #
        # # 删除，然后在进行重新设置是因为他是一个 parameter 并且 requires grad
        # delattr(self.module, self.name)
        # # del self.module._parameters[self.name]  # 是一个 OrderedDict，将我们的 weight 删除
        #
        # setattr(self.module, self.name + '_u', u)
        # setattr(self.module, self.name + '_v', v)
        # setattr(self.module, self.name + '_w_hat', w_hat)

        # print('this is the test1 to show parameters')

    def _made_u_v_w(self):
        try:
            u = getattr(self.module, self.name + '_u')
            v = getattr(self.module, self.name + '_v')
            w_hat = getattr(self.module, self.name + '_w_hat')
            return True
        except AttributeError:
            return False

    def _update_u_v(self, eps=1e-12):
        # 获取我们的属性值
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w_hat = getattr(self.module, self.name + '_w_hat')

        # 这里加这个类型转换是因为 可能 u, v 收到 auto case 的影响，编程 float16了，
        # 但是在sample_images时候，并没有变，导致类型错误
        u = u.to(w_hat.dtype)
        v = v.to(w_hat.dtype)
        # 进行迭代，计算谱范数
        height = w_hat.data.shape[0]
        for _ in range(self.power_iterations):  # 这里很有可能直接把你的类型给改了
            u.data = l2normalize(torch.mv(w_hat.view(height, -1), v.data))
            v.data = l2normalize(torch.mv(w_hat.view(height, -1).T, u.data))
        sigma = torch.dot(u.data, torch.mv(w_hat.view(height, -1), v.data))

        # return w_hat / (sigma + eps)
        setattr(self.module, self.name, w_hat / (sigma + eps))

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SelfAttention(nn.Module):
    """
    自注意力机制层，直接按照那个 self-attention 图进行写就好了。
    不过
    """
    def __init__(self, in_dim, rate=16, activation=None):
        super(SelfAttention, self).__init__()
        # 这个 activation 好像并没有用到，应为这个 conv 只是为了减少我们 self-attention 计算量
        self.act = activation if activation is not None else nn.ReLU()
        # 首先是 q, k, v 的三个线性变化层，我们这里可以将 v 不从 c -> c_hat, 这样我们后面就不需要再将输出转换了
        # 这里的 Conv 可以试着用 SpectralNorm，也可以不用，可以都试试。不过因为他仅仅是最后两层，效果不很会明显
        self.conv_q = nn.Conv2d(in_dim, in_dim // rate, kernel_size=1)
        self.conv_k = nn.Conv2d(in_dim, in_dim // rate, kernel_size=1)
        self.conv_v = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(.0))     # 初始化为0 是刚刚开始不进行 self-attention，仅仅学习局部的信息

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        queries = self.conv_q(x).reshape(batch_size, -1, height * width)
        keys = self.conv_k(x).reshape(batch_size, -1, height * width)
        values = self.conv_v(x).reshape(batch_size, -1, height * width)
        weight = torch.softmax(torch.bmm(queries.permute(0, 2, 1), keys), dim=-1)

        # 正常 self attention 写法
        # y = torch.bmm(weight, values.permute(0, 2, 1)).permute(0, 2, 1)
        y = torch.bmm(values, weight.permute(0, 2, 1)).reshape(batch_size, -1, height, width)
        return x + self.gamma * y


def test_spectral_norm():
    device = torch.device('cuda')
    conv = nn.Conv2d(2, 3, 4, stride=2, padding=1)
    spectral_norm = SpectralNorm(conv).to(device)
    # spectral_norm = nn.parallel.DataParallel(SpectralNorm(conv), device_ids=config.DEVICES)
    inputs = torch.rand((64, 2, 64, 64)).to(torch.device('cuda'))
    outputs = spectral_norm(inputs)
    print(inputs.shape)
    print(outputs.shape)

    # conv = nn.ConvTranspose2d(2, 3, 4, stride=2, padding=1)
    # spectral_norm = SpectralNorm(conv)
    # inputs = torch.rand((64, 2, 64, 64))
    # outputs = spectral_norm(inputs)
    # print(inputs.shape)
    # print(outputs.shape)


def test_attention():
    input = torch.rand((64, 512, 64, 64))
    attention = SelfAttention(512)
    output = attention(input)
    print(output.shape)
    # print(weight.shape)


if __name__ == '__main__':
    test_spectral_norm()
