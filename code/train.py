""" Training of ProGAN using WGAN-GP loss"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import random
from math import log2

from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
)
from dataset import (
    get_data_iter
)
from model import StyleDiscriminator, StyleGenerator
import config


def code_init():
    """
        主要是开启加速，并且查看是否需要创建文件夹
        这里并且加入了固定随机种子的操作
    """
    # 加速
    torch.backends.cudnn.benchmarks = True

    # 随机种子的设置
    # Set random seem for reproducibility，主要是方面我们后期的可重复性
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # 文件夹的创建
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_SAVE_DIR, exist_ok=True)
    os.makedirs(config.IMAGE_SAVE_DIR, exist_ok=True)


def train_fn(
        cur_stage,
        critic,
        gen,
        loader,
        dataset,
        step,
        alpha,
        opt_critic,
        opt_gen,
        tensorboard_step,
        writer,
        scheduler_gen,
        scheduler_critic,
        scaler_gen,
        scaler_critic,
):
    # from tqdm import tqdm
    # loop = tqdm(loader, leave=True, total=len(loader))
    loop = loader

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)] <-> min -E[critic(real)] + E[critic(fake)]
        # which is equivalent to minimizing the negative of the expression
        noise = torch.randn(cur_batch_size, config.DLATENT_SIZE).to(config.DEVICE)

        with torch.cuda.amp.autocast():
            fake = gen(noise, alpha, step)
            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)
            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + config.LAMBDA_GP * gp
                    + (0.001 * torch.mean(critic_real ** 2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)
            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
                (config.PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        if batch_idx % config.SAMPLE_INTERVAL == 0:
            with torch.no_grad():
                # 这个选好一点，否则别出现了 gpu 显存不够的情况
                max_batch_size = config.BATCH_SIZE[cur_stage]
                fixed_fakes = gen(config.FIXED_NOISE[:max_batch_size], alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                cur_stage,
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

        # loop.set_postfix(
        #     gp=gp.item(),
        #     loss_critic=loss_critic.item(),
        # )
    scheduler_gen.step()
    scheduler_critic.step()

    return tensorboard_step, alpha


def main():
    # 注意，我们的模型数据的加载，需要在 progressive 的过程中进行
    # 定义网络
    generator = StyleGenerator(
        mapping_fmaps=config.MAPPING_FMAPS,
        dlatent_size=config.DLATENT_SIZE,
        resolution=config.RESOLUTION,
        fmap_base=config.FMAP_BASE,
        fmap_max=config.FMAP_MAX,
        fmap_decay=config.FMAP_DECAY,
        num_channels=config.NUM_CHANNELS,
        use_wscale=True,
        use_pixel_norm=True,
        use_instance_norm=True,
        use_noise=True,
        use_style=True
    ).to(config.DEVICE)
    discriminator = StyleDiscriminator(
        resolution=config.RESOLUTION,
        fmap_base=config.FMAP_BASE,
        fmap_max=config.FMAP_MAX,
        fmap_decay=config.FMAP_DECAY,
        num_channels=config.NUM_CHANNELS
    ).to(config.DEVICE)

    # Optimizer 和 Scheduler 可以考虑再加入 torch.cuda.amp.Grad_Scaler
    # ExponentialLR 指数衰减学习率
    # new_lr=init_lr * gamma ** epoch
    optim_disc = optim.Adam(discriminator.parameters(), lr=config.LR_D,
                            betas=(config.BETA1, config.BETA2))
    optim_gen = optim.Adam(generator.parameters(), lr=config.LR_G,
                           betas=(config.BETA1, config.BETA2))
    # scheduler_D = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=config.SCHEDULER_GAMMA)
    # scheduler_G = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=config.SCHEDULER_GAMMA)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optim_disc, T_max=config.T_MAX)
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optim_gen, T_max=config.T_MAX)
    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()

    # 预训练加载权重
    if config.LOAD_MODEL:
        model_path_gen = os.path.join(config.MODEL_SAVE_DIR, f'gen-{config.SAVE_MODEL_INDEX}.pth.tar')
        model_path_disc = os.path.join(config.MODEL_SAVE_DIR, f'disc-{config.SAVE_MODEL_INDEX}.pth.tar')
        if os.path.exists(model_path_gen):
            load_checkpoint(
                model_path_gen,
                generator,
                optim_gen,
                config.LR_G
            )
            load_checkpoint(
                model_path_disc,
                discriminator,
                optim_disc,
                config.LR_D
            )
        else:
            print('======== warning!! ========')
            print('load model file does not exist!')

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # for tensorboard plotting
    writer = SummaryWriter(config.LOG_SAVE_DIR)

    generator.train()
    discriminator.train()

    tensorboard_step = 0
    # start at step that corresponds to img size that we set in config
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / config.BASE_CONSTANT_IMAGE_SIZE))
    end_step = int(log2(config.RESOLUTION / config.BASE_CONSTANT_IMAGE_SIZE))
    for cur_stage, num_epochs in enumerate(config.PROGRESSIVE_EPOCHS[step:end_step+1], start=step):
        alpha = 1e-5  # start with very low alpha
        dataset, loader = get_data_iter(
            batch_size=config.BATCH_SIZE[cur_stage],        # 注意这个 batch_size 最好设置为随着阶段变化的
            is_train=True,
            img_size=config.BASE_CONSTANT_IMAGE_SIZE * (2 ** step),   # 4->0, 8->1, 16->2, 32->3, 64 -> 4
            num_channels=config.NUM_CHANNELS,
            num_workers=config.NUM_WORKERS,
            max_num=config.DATA_SET_MAX_NUM,
            is_gray=(config.NUM_CHANNELS == 1)      # 是否使用灰度图
        )
        print(f"Current image size: {config.BASE_CONSTANT_IMAGE_SIZE * 2 ** step}, "
              f"batch_size: {config.BATCH_SIZE[cur_stage]}")

        # 内部循环进行训练
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            tensorboard_step, alpha = train_fn(
                cur_stage,
                discriminator,
                generator,
                loader,
                dataset,
                step,
                alpha,
                opt_critic=optim_disc,
                opt_gen=optim_gen,
                tensorboard_step=tensorboard_step,
                writer=writer,
                scheduler_gen=scheduler_G,
                scheduler_critic=scheduler_D,
                scaler_gen=scaler_G,
                scaler_critic=scaler_D,
            )

            # 模型的保存
            if config.SAVE_MODEL:
                cur_save_index = epoch
                if cur_stage > 0:
                    for _ in config.PROGRESSIVE_EPOCHS[:cur_stage-1]:
                        cur_save_index += _
                if cur_save_index % config.SAVE_MODEL_INTERVAL == 0:
                    cur_save_index //= config.SAVE_MODEL_INTERVAL
                    model_path_gen = os.path.join(config.MODEL_SAVE_DIR, f'gen-{cur_save_index}.pth.tar')
                    model_path_disc = os.path.join(config.MODEL_SAVE_DIR, f'disc-{cur_save_index}.pth.tar')
                    save_checkpoint(generator, optim_gen, filename=model_path_gen)
                    save_checkpoint(discriminator, optim_disc, filename=model_path_disc)

        step += 1  # progress to the next img size


if __name__ == "__main__":
    code_init()
    main()
