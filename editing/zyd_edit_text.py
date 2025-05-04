import argparse
import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from DIFCLIP.losses.clip_loss import CLIPLoss
from DIFCLIP.losses import id_loss
# from DIFCLIP.utils.diffusion_utils import get_beta_schedule, denoising_step
"""
Reference code:
https://github.com/justinpinkney/stable-diffusion/blob/main/notebooks/imagic.ipynb
"""

parser = argparse.ArgumentParser()

# directories
parser.add_argument('--config', type=str, default='../configs/512_text.yaml')
parser.add_argument('--ckpt', type=str, default='/home/ww/4T/zyd/Collaborative-Diffusion_10/outputs/512_text/2024-11-11T02-01-02_512_text/checkpoints/epoch=000188.ckpt')
parser.add_argument('--save_folder', type=str, default='outputs/my_edit/Ablation')
parser.add_argument(
    '--input_image_path',
    type=str,
    default='/home/ww/4T/zyd/Collaborative-Diffusion_10/editing/text_editing_input/Ablation/6263.jpg')
parser.add_argument(
    '--text_prompt',
    type=str,
    default='这位年轻女士没有刘海，没有眼镜，看起来很严肃，脸上没有笑容.'
)
# hyperparameters
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--stage1_lr', type=float, default=0.001)
parser.add_argument('--stage1_num_iter', type=int, default=500)
parser.add_argument('--stage2_lr', type=float, default=1e-6)
parser.add_argument('--stage2_num_iter', type=int, default=1000)
parser.add_argument(
    '--alpha_list',
    type=str,
    default='-1, -0.5, 0, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5')
parser.add_argument('--set_random_seed', type=bool, default=False)
parser.add_argument('--save_checkpoint', type=bool, default=True)

args = parser.parse_args()


def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


@torch.no_grad()
def sample_model(model,
                 sampler,
                 c,
                 h,
                 w,
                 ddim_steps,
                 scale,
                 ddim_eta,
                 start_code=None,
                 n_samples=1):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])

    # print(f'model.model.parameters(): {model.model.parameters()}')
    # for name, param in model.model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data)
    #         break

    # print(f'unconditional_guidance_scale: {scale}') # 1.0
    # print(f'unconditional_conditioning: {uc}') # None
    with model.ema_scope("Plotting"):

        shape = [3, 64, 64]  # [4, h // 8, w // 8]
        samples_ddim, _ = sampler.sample(
            S=ddim_steps,
            conditioning=c,
            batch_size=n_samples,
            shape=shape,
            verbose=False,
            start_code=start_code,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
        )
        return samples_ddim


def load_img(path, target_size=256):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose([
        # transforms.Resize(target_size),
        # transforms.CenterCrop(target_size),
        transforms.ToTensor(),
    ])
    image = tform(image)
    return 2. * image - 1.


def decode_to_im(samples, n_samples=1, nrow=1):
    """Decode a latent and return PIL image"""
    samples = model.decode_first_stage(samples)
    ims = torch.clamp((samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_sample = 255. * rearrange(
        ims.cpu().numpy(),
        '(n1 n2) c h w -> (n1 h) (n2 w) c',
        n1=n_samples // nrow,
        n2=nrow)
    return Image.fromarray(x_sample.astype(np.uint8))


if __name__ == '__main__':

    args.alpha_list = [float(i) for i in args.alpha_list.split(',')]

    device = "cuda"  # "cuda:0"

    # Generation parameters  生成参数
    scale = 1.0
    h = 256
    w = 256
    ddim_steps = 50
    ddim_eta = 1.0

    # initialize model  模型初始化
    global_model = load_model_from_config(args.config, args.ckpt, device)

    input_image = args.input_image_path
    image_name = input_image.split('/')[-1]

    # prompt = '这个人看起来很严肃，脸上没有笑容，没有眼镜，也没有刘海。她已步入中年。'
    prompt = args.text_prompt

    torch.manual_seed(args.seed)

    model = copy.deepcopy(global_model)
    sampler = DDIMSampler(model)

    # prepare directories  目录
    save_dir = os.path.join(args.save_folder, image_name, "inter_idloss", str(prompt)[:10])
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(
        f'================================================================================'
    )
    print(f'input_image: {input_image} | text: {prompt}')

    # read input image  读数据
    init_image = load_img(input_image).to(device).unsqueeze(
        0)  # [1, 3, 256, 256]      在加载的图像张量维度前添加一个维度，将其转换为形状为(1, C, H, W)的张量，其中C是通道数，H和W是图像的高度和宽度。
    gaussian_distribution = model.encode_first_stage(init_image)
    init_latent = model.get_first_stage_encoding(
        gaussian_distribution)  # [1, 3, 64, 64]
    img = decode_to_im(init_latent)
    print("img:", type(img))
    img.save(os.path.join(save_dir, 'input_image_reconstructed.png'))    #  重建

    # obtain text embedding
    emb_tgt = model.get_learned_conditioning([prompt])
    print("emb_tgt.shape:", emb_tgt.shape)
    emb_ = emb_tgt.clone()
    torch.save(emb_, os.path.join(save_dir, 'emb_tgt.pt'))
    emb = torch.load(os.path.join(save_dir, 'emb_tgt.pt'))  # [1, 77, 640]   嵌入文本

    # Sample the model with a fixed code to see what it looks like  使用固定代码对模型进行采样，看看它是什么样子
    quick_sample = lambda x, s, code: decode_to_im(
        sample_model(
            model, sampler, x, h, w, ddim_steps, s, ddim_eta, start_code=code))
    # start_code = torch.randn_like(init_latent)
    start_code = torch.randn((1, 3, 64, 64), device=device)
    # #ddim_invertion
    # img = Image.open(args.input_image_path).convert("RGB")
    # img = img.resize((512, 512), Image.LANCZOS)
    # img = np.array(img) / 255
    # img = torch.from_numpy(img).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(dim=0)
    # img = img.to(device)
    # torchvision.utils.save_image(img, os.path.join("outputs/my_edit", f'0_orig.png'))
    # x0 = (img - 0.5) * 2.
    # betas = get_beta_schedule(
    #     beta_start=0.0001,
    #     beta_end=0.02,
    #     num_diffusion_timesteps=1000
    # )
    # betas = torch.from_numpy(betas).float().to(device)
    # alphas = 1.0 - betas
    # alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)
    # alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    # posterior_variance = betas.cpu() * \
    #                      (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # logvar = np.log(np.maximum(posterior_variance, 1e-20))
    #
    # with torch.no_grad():
    #     seq_inv = np.linspace(0, 1, 40) * 400
    #     seq_inv = [int(s) for s in list(seq_inv)]
    #     seq_inv_next = [-1] + list(seq_inv[:-1])
    #     x = x0.clone()
    #     with tqdm(total=len(seq_inv), desc=f"Inversion process") as progress_bar:
    #         for it, (i, j) in enumerate(zip((seq_inv_next[1:]), (seq_inv[1:]))):
    #             t = (torch.ones(1) * i).to(device)
    #             t_prev = (torch.ones(1) * j).to(device)
    #
    #             x = denoising_step(x, t=t, t_next=t_prev, models=model,
    #                                logvars=logvar,
    #                                sampling_type='ddim',
    #                                b=betas,
    #                                eta=0,
    #                                learn_sigma=True)
    #
    #             progress_bar.update(1)
    #
    #     x_lat0 = x.clone()
    #     torchvision.utils.save_image((x_lat0 + 1) * 0.5, os.path.join("outputs/my_edit",
    #                                                     f'1_lat_ninv{40}.png'))
    #
    #
    #
    #
    #
    #
    #
    #


    torch.save(start_code, os.path.join(save_dir,
                                        'start_code.pt'))  # [1, 3, 64, 64]
    torch.manual_seed(args.seed)
    img = quick_sample(emb_tgt, scale, start_code)          #   基于文本生成个图像
    img_0 = img
    img.save(os.path.join(save_dir, 'A_start_tgtText_origDM.png'))
    clip_model_name = "ViT-B/32"
    clip_loss_func = CLIPLoss(
        device,
        lambda_direction=1,
        lambda_patch=0,
        lambda_global=0,
        lambda_manifold=0,
        lambda_texture=0,
        clip_model= clip_model_name)
    id_loss_func = id_loss.IDLoss().to(device).eval()

    # ======================= (A) Text Embedding Optimization ===================================
    print('########### Step 1 - Optimise the embedding ###########')
    emb.requires_grad = True        #   将emb参数设置为需要进行梯度计算，即将其设置为可训练参数。
    opt = torch.optim.Adam([emb], lr=args.stage1_lr)    #   创建一个Adam优化器，用于优化emb参数。args.stage1_lr表示学习率
    criteria = torch.nn.MSELoss()       #   定义均方误差损失函数作为优化目标。

    history = []

    pbar = tqdm(range(args.stage1_num_iter))    #   创建一个进度条迭代器，用于追踪训练的迭代次数。
    emb_before = emb.clone()
    for i in pbar:
        opt.zero_grad()

        if args.set_random_seed:
            torch.seed()
        noise = torch.randn_like(init_latent)   #   生成与init_latent形状相同的随机噪声。
        t_enc = torch.randint(1000, (1, ), device=device)       #   生成一个随机整数作为时间步长。  随机步长
        z = model.q_sample(init_latent, t_enc, noise=noise)     #   通过模型的q_sample方法生成编码样本z。
        # print(f"Z：{z.shape}, t_enc: {t_enc.shape}, emb: {emb.shape}")

        pred_noise = model.apply_model(z, t_enc, emb)           #   使用模型的apply_model方法根据编码样本z和emb参数生成噪声样本的预测

        loss = criteria(pred_noise, noise)                      #   计算预测噪声和真实噪声之间的均方误差损失。
        loss.backward()         #   反向传播计算梯度
        pbar.set_postfix({"loss": loss.item()})         #   更新进度条显示的损失值。
        history.append(loss.item())         #   将当前损失值添加到history列表中。
        opt.step()

    emb_after = emb.clone()
    emb_change = (emb_after - emb_before).norm()
    print(f"Difference in emb: {emb_change}")
    plt.plot(history)
    plt.show()
    torch.save(emb, os.path.join(save_dir, 'emb_opt.pt'))
    emb_opt = torch.load(os.path.join(save_dir, 'emb_opt.pt'))  # [1, 77, 640]

    torch.manual_seed(args.seed)
    img = quick_sample(emb_opt, scale, start_code)
    img.save(os.path.join(save_dir, 'A_end_optText_origDM.png'))        #   优化后的文本嵌入生成的图像

    # #Interpolate the embedding 使用不同的alpha值对目标emb_tgt和优化后的emb_opt进行线性插值，并生成插值后的图像样本。让我们逐行解释代码的功能
    # for idx, alpha in enumerate(args.alpha_list):
    #     print(f'alpha={alpha}')
    #     new_emb = alpha * emb_tgt + (1 - alpha) * emb_opt
    #     torch.manual_seed(args.seed)
    #     img = quick_sample(new_emb, scale, start_code)
    #     img.save(
    #         os.path.join(
    #             save_dir,
    #             f'0A_interText_origDM_{idx}_alpha={round(alpha,3)}.png'))

    # ======================= (B) Model Fine-Tuning ===================================
    print('########### Step 2 - Fine tune the model ###########')   #   优化模型的参数emb_opt，并生成优化后的图像样本。训练过程使用Adam优化器和均方误差损失函数，将模型设置为训练模式进行参数更新，然后将模型设置为评估模式生成图像样本。如果需要，还可以保存优化后的参数和模型的检查点。
    emb_opt.requires_grad = False
    emb_opt_B2 = emb_opt.repeat(2, 1, 1)
    init_latent_B2 = init_latent.repeat(2, 1, 1, 1)
    model.train()

    opt = torch.optim.Adam(model.model.parameters(), lr=args.stage2_lr)
    criteria = torch.nn.MSELoss()
    history = []

    pbar = tqdm(range(args.stage2_num_iter))
    for i in pbar:
        opt.zero_grad()

        if args.set_random_seed:
            torch.seed()
        noise = torch.randn_like(init_latent_B2)
        t_enc = torch.randint(model.num_timesteps, (1, ), device=device)
        z = model.q_sample(init_latent_B2, t_enc, noise=noise)     #   真实图像
        # print(f"Z：{z.shape}, t_enc: {t_enc.shape}, emb: {emb.shape}")
        pred_noise = model.apply_model(z, t_enc, emb_opt_B2)       #   预测噪声

        loss = criteria(pred_noise, noise)                      #   噪声损失
        # loss_id = torch.mean(id_loss_func(init_latent, z))
        # print(f"{i}loss_id:", loss_id)
        # loss = loss_MSE*0.5 + loss_id*0.5
        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

    model.eval()
    plt.plot(history)
    plt.show()
    torch.manual_seed(args.seed)
    img = quick_sample(emb_opt, scale, start_code)              #   emb优化后采样生成图像
    img.save(os.path.join(save_dir, 'B_end_optText_optDM.png'))
    # Should look like the original image

    if args.save_checkpoint:
        ckpt = {
            "state_dict": model.state_dict(),
        }
        ckpt_path = os.path.join(save_dir, 'optDM.ckpt')
        print(f'Saving optDM to {ckpt_path}')
        torch.save(ckpt, ckpt_path)

    for idx, alpha in enumerate(args.alpha_list):
        print(f'alpha={alpha}')
        new_emb = alpha * emb_tgt + (1 - alpha) * emb_opt
        torch.manual_seed(args.seed)
        img = quick_sample(new_emb, scale, start_code)
        img.save(
            os.path.join(
                save_dir,
                f'intermediate_{idx}_alpha={round(alpha,3)}.png'))        #   对优化后的文本嵌入进行插值，得到最后的图像

    #ID—loss
    emb = emb_opt
    emb.requires_grad = True
    opt = torch.optim.Adam([emb], lr=args.stage1_lr)
    # emb_opt.requires_grad = False
    model.train()
    new_emb = 0.6 * emb_tgt + (1 - 0.6) * emb
    new_emb_B2 = new_emb.repeat(2, 1, 1)
    start_code_B2 = start_code.repeat(2, 1, 1, 1)
    # opt = torch.optim.Adam(model.model.parameters(), lr=0.0000001)
    history = []
    img0_array = np.array(img_0) / 255.0
    img0_tensor = torch.from_numpy(img0_array).float()
    img0_tensor = img0_tensor.permute(2, 0, 1)
    img0_tensor = img0_tensor.unsqueeze(0).to(device)
    ddim_sampler = DDIMSampler(model)

    pbar = tqdm(range(500))
    for i in pbar:
        opt.zero_grad()

        z_0_batch, intermediates = ddim_sampler.sample(
            S=50,
            batch_size=start_code_B2.shape[0],
            shape=(3, 64, 64),
            conditioning=new_emb_B2,
            verbose=False,
            eta=1.0,
            log_every_t=1)
        x_0_batch = model.decode_first_stage(z_0_batch)
        x_0 = x_0_batch[1, :, :, :].unsqueeze(0)  # [1, 3, 256, 256]
        x_0 = x_0.permute(0, 2, 3, 1).to('cpu').numpy()
        x_0 = (x_0 + 1.0) * 127.5
        np.clip(x_0, 0, 255, out=x_0)  # clip to range 0 to 255
        x_0 = x_0.astype(np.uint8)
        x_0 = Image.fromarray(x_0[0])

        print(type(x_0))
        img_array = np.array(x_0) / 255.0
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        loss = torch.mean(id_loss_func(img0_tensor, img_tensor))

        loss.backward()
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())
        opt.step()

    model.eval()
    plt.plot(history)
    plt.show()
    torch.save(emb, os.path.join(save_dir, 'emb_IDLoss.pt'))
    emb_IDLoss = torch.load(os.path.join(save_dir, 'emb_IDLoss.pt'))


    # Should look like the original image

    # if args.save_checkpoint:
    #     ckpt = {
    #         "state_dict": model.state_dict(),
    #     }
    #     ckpt_path = os.path.join(save_dir, 'optDM_Idloss.ckpt')
    #     print(f'Saving optDM to {ckpt_path}')
    #     torch.save(ckpt, ckpt_path)


    print('########### Step 3 - Generate images ###########')       #   使用不同的alpha值对目标emb_tgt和优化后的emb_opt进行线性插值，并生成插值后的图像样本
    # Interpolate the embedding
    for idx, alpha in enumerate(args.alpha_list):
        print(f'alpha={alpha}')
        new_emb = alpha * emb_tgt + (1 - alpha) * emb_IDLoss
        torch.manual_seed(args.seed)
        img = quick_sample(new_emb, scale, start_code)
        # img.save(
        #     os.path.join(
        #         save_dir,
        #         f'0C_interText_optDM_{idx}_alpha={round(alpha,3)}.png'))        #   对优化后的文本嵌入进行插值，得到最后的图像
        img.save(os.path.join(save_dir,  f'final_{idx}_alpha={round(alpha,3)}.png'))

    print('Done')
