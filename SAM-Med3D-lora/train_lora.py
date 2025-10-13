import argparse
import datetime
import logging
import os
import random
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchio as tio
from monai.losses import DiceCELoss, FocalLoss, DiceFocalLoss, HausdorffDTLoss
from torch.backends import cudnn
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

#from skimage.morphology import skeletonize_3d
from segment_anything.build_sam3D import sam_model_registry3D
from utils.click_method import get_next_click3D_torch_2
#from utils.click_method import get_points_on_skeleton_with_negatives
from utils.data_loader import Dataset_Union_ALL, Union_Dataloader, Dataset_Union_ALL_Val
from utils.data_paths import img_datas
import torch.nn as nn

join = os.path.join

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', type=str, default='union_train')
parser.add_argument('--click_type', type=str, default='random')
parser.add_argument('--multi_click', action='store_true', default=False)
parser.add_argument('--model_type', type=str, default='vit_b_ori')
parser.add_argument('--checkpoint', type=str, default='ckpt/sam_med3d.pth')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--work_dir', type=str, default='work_dir')

# train
parser.add_argument('--num_workers', type=int, default=24)
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0, 1])
parser.add_argument('--multi_gpu', action='store_true', default=False)
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--allow_partial_weight', action='store_true', default=False)

# lr_scheduler
parser.add_argument('--lr_scheduler', type=str, default='multisteplr')
parser.add_argument('--step_size', type=list, default=[120, 180])
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--accumulation_steps', type=int, default=20)
parser.add_argument('--lr', type=float, default=8e-4)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--port', type=int, default=12361)
parser.add_argument('--use_lora', action='store_true', default=True, help='Abilita LoRA nel modello')
parser.add_argument('--lora_r', type=int, default=12, help='Rank delle matrici LoRA (A e B)')


args = parser.parse_args()

device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in args.gpu_ids])
logger = logging.getLogger(__name__)
LOG_OUT_DIR = join(args.work_dir, args.task_name)
click_methods = {
    'random': get_next_click3D_torch_2,
}
MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)



class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        # Freeze pretrained weights
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.scaling * self.lora_up(self.lora_down(x))

class LoRA_qkv(nn.Module):
    def __init__(self, qkv_layer, rank=4, alpha=1.0):
        super().__init__()
        assert isinstance(qkv_layer, nn.Linear)
        in_features = qkv_layer.in_features
        out_features = qkv_layer.out_features
        self.lora = LoRALinear(in_features, out_features, rank=rank, alpha=alpha)

    def forward(self, x):
        return self.lora(x)


class LoRA_image_encoder(nn.Module):
    def __init__(self, image_encoder, rank=4, alpha=1.0):
        super().__init__()
        self.image_encoder = image_encoder

        for i, blk in enumerate(self.image_encoder.blocks):
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'qkv') and isinstance(blk.attn.qkv, nn.Linear):
                print(f"[LoRA] Replacing attn.qkv in block {i}")
                blk.attn.qkv = LoRA_qkv(blk.attn.qkv, rank=rank, alpha=alpha)
            else:
                print(f"[LoRA] Warning: cannot replace attn.qkv in block {i}")

    def forward(self, x):
        return self.image_encoder(x)



MODEL_SAVE_PATH = join(args.work_dir, args.task_name)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


def build_model(args):
    sam_model = sam_model_registry3D[args.model_type](checkpoint=None)
    print("SAM model built")

    if args.checkpoint is not None:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        sam_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        base_dtype = next(sam_model.image_encoder.parameters()).dtype
        print(f"📌 SAM Image Encoder base dtype: {base_dtype}")

    if args.use_lora:
        print(f"USING LORA image encoder with rank {args.lora_r}")
        sam_model.image_encoder = LoRA_image_encoder(sam_model.image_encoder, rank=args.lora_r)

        sam_model.image_encoder = sam_model.image_encoder.to(dtype=base_dtype)
        for name, param in sam_model.named_parameters():
              print(f"❗ Param '{name}' has dtype {param.dtype}")
        sam_model = sam_model.to(device=device, dtype=base_dtype)


    return sam_model


def get_dataloaders(args):
    train_dataset = Dataset_Union_ALL(
        paths=img_datas,
        data_type="Tr",
        transform=tio.Compose([
            tio.ToCanonical(),
            tio.CropOrPad(
                mask_name='label',
                target_shape=(args.img_size, args.img_size, args.img_size)
            ),
            tio.RandomFlip(axes=(0, 1, 2)),

        ]),
        threshold=10  # ✅ questo è il posto giusto!
    )


    val_dataset = Dataset_Union_ALL(
    paths=img_datas,
    data_type="Val",
    transform=tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(
                mask_name='label',
                target_shape=(args.img_size, args.img_size, args.img_size)
            ),

        ]),
    threshold=10)

    if args.multi_gpu:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        shuffle = False
    else:
        train_sampler = None
        val_sampler = None

        shuffle = True

    # train_dataloader = tio.SubjectsLoader(
    train_dataloader = Union_Dataloader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataloader = Union_Dataloader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_dataloader, val_dataloader


class BaseTrainer:

    def __init__(self, model, dataloaders, args):

        self.model = model
        self.train_dataloader = dataloaders[0]
        self.val_dataloader = dataloaders[1]
        self.args = args
        self.best_loss = np.inf
        self.best_dice = 0.0
        self.step_best_loss = np.inf
        self.step_best_dice = 0.0
        self.losses = []
        self.dices = []
        self.ious = []
        self.val_losses = []
        self.val_dices = []
        self.set_loss_fn()
        self.set_optimizer()
        self.set_lr_scheduler()
        if (args.resume):
            self.init_checkpoint(
                join(self.args.work_dir, self.args.task_name, 'sam_model_latest.pth'))
        else:
            self.start_epoch = 0

        self.norm_transform = tio.ZNormalization(masking_method=lambda x: x > 0)

    def set_loss_fn(self):

        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def set_optimizer(self):
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        # ❄️ Congela tutto
        for param in sam_model.parameters():
            param.requires_grad = False

        for name, param in sam_model.image_encoder.named_parameters():
            if 'lora_down' in name or 'lora_up' in name:
                param.requires_grad = True

        # ✅ Sblocca tutto nel prompt_encoder
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = True

        # ✅ Sblocca tutto nel mask_decoder
        for param in sam_model.mask_decoder.parameters():
            param.requires_grad = True
        for name, param in sam_model.named_parameters():
            if param.requires_grad:
                print(f"✅ {name} | shape: {tuple(param.shape)}")
        self.optimizer = torch.optim.AdamW(
            [
                {
                    'params': sam_model.image_encoder.parameters()
                },  # , 'lr': self.args.lr * 0.1},
                {
                    'params': sam_model.prompt_encoder.parameters(),
                    'lr': self.args.lr * 0.1
                },
                {
                    'params': sam_model.mask_decoder.parameters(),
                    'lr': self.args.lr * 0.1
                },
            ],
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay)




    def set_lr_scheduler(self):
        if self.args.lr_scheduler == "multisteplr":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     self.args.step_size,
                                                                     self.args.gamma)
        elif self.args.lr_scheduler == "steplr":
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                self.args.step_size[0],
                                                                self.args.gamma)
        elif self.args.lr_scheduler == 'coswarm':
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, 0.1)

    def init_checkpoint(self, ckp_path):
        last_ckpt = None
        if os.path.exists(ckp_path):
            if self.args.multi_gpu:
                dist.barrier()
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)
            else:
                last_ckpt = torch.load(ckp_path, map_location=self.args.device, weights_only=False)

        if last_ckpt:
            if (self.args.allow_partial_weight):
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'], strict=False)
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'], strict=False)
            else:
                if self.args.multi_gpu:
                    self.model.module.load_state_dict(last_ckpt['model_state_dict'])
                else:
                    self.model.load_state_dict(last_ckpt['model_state_dict'])
            if not self.args.resume:
                self.start_epoch = 0
            else:
                self.start_epoch = last_ckpt['epoch']
                self.optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
                self.lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
                self.losses = last_ckpt['losses']
                self.dices = last_ckpt['dices']
                self.best_loss = last_ckpt['best_loss']
                self.best_dice = last_ckpt['best_dice']
            print(f"Loaded checkpoint from {ckp_path} (epoch {self.start_epoch})")
        else:
            self.start_epoch = 0
            print(f"No checkpoint found at {ckp_path}, start training from scratch")

    def save_checkpoint(self, epoch, state_dict, describe="last"):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "losses": self.losses,
                "dices": self.dices,
                "best_loss": self.best_loss,
                "best_dice": self.best_dice,
                "args": self.args,
                "used_datas": img_datas,
            }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

    def batch_forward(self, sam_model, image_embedding, gt3D, low_res_masks, points=None):

        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=points,
            boxes=None,
            masks=low_res_masks,
        )
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding.to(device),  # (B, 256, 64, 64)
            image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res_masks,
                                   size=gt3D.shape[-3:],
                                   mode='trilinear',
                                   align_corners=False)
        return low_res_masks, prev_masks

    def get_points(self, prev_masks, gt3D):
        batch_points, batch_labels = click_methods[self.args.click_type](prev_masks, gt3D)

        points_co = torch.cat(batch_points, dim=0).to(device)
        points_la = torch.cat(batch_labels, dim=0).to(device)

        self.click_points.append(points_co)
        self.click_labels.append(points_la)

        points_multi = torch.cat(self.click_points, dim=1).to(device)
        labels_multi = torch.cat(self.click_labels, dim=1).to(device)

        if self.args.multi_click:
            points_input = points_multi
            labels_input = labels_multi
        else:
            points_input = points_co
            labels_input = points_la
        return points_input, labels_input

    def interaction(self, sam_model, image_embedding, gt3D, num_clicks):
        return_loss = 0
        prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
        low_res_masks = F.interpolate(prev_masks.float(),
                                      size=(args.img_size // 4, args.img_size // 4,
                                            args.img_size // 4))
        random_insert = np.random.randint(2, 9)
        for num_click in range(num_clicks):
            points_input, labels_input = self.get_points(prev_masks, gt3D)

            if num_click == random_insert or num_click == num_clicks - 1:
                low_res_masks, prev_masks = self.batch_forward(sam_model,
                                                               image_embedding,
                                                               gt3D,
                                                               low_res_masks,
                                                               points=None)
            else:
                low_res_masks, prev_masks = self.batch_forward(sam_model,
                                                               image_embedding,
                                                               gt3D,
                                                               low_res_masks,
                                                               points=[points_input, labels_input])
            loss = self.seg_loss(prev_masks, gt3D)
            return_loss += loss
        return prev_masks, return_loss

    def get_dice_score(self, prev_masks, gt3D):

        def compute_dice(mask_pred, mask_gt):
            mask_threshold = 0.5

            mask_pred = (mask_pred > mask_threshold)
            mask_gt = (mask_gt > 0)

            volume_sum = mask_gt.sum() + mask_pred.sum()
            if volume_sum == 0:
                return np.NaN
            volume_intersect = (mask_gt & mask_pred).sum()
            return 2 * volume_intersect / volume_sum

        pred_masks = (prev_masks > 0.5)
        true_masks = (gt3D > 0)
        dice_list = []
        for i in range(true_masks.shape[0]):
            dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
        return (sum(dice_list) / len(dice_list)).item()


    def train_epoch(self, epoch, num_clicks):
        epoch_loss = 0
        epoch_iou = 0
        self.model.train()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model
            self.args.rank = -1

        if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
            tbar = tqdm(self.train_dataloader)
        else:
            tbar = self.train_dataloader

        self.optimizer.zero_grad()
        step_loss = 0
        epoch_dice = 0
        for step, data3D in enumerate(tbar):
            try:
                image3D, gt3D = data3D["image"], data3D["label"]
            except Exception as e:
                print(f"Error processing batch at step {step}: {e}")
            # import pdb; pdb.set_trace()
            my_context = self.model.no_sync if self.args.rank != - \
                1 and step % self.args.accumulation_steps != 0 else nullcontext

            with my_context():

                image3D = self.norm_transform(image3D.squeeze(dim=1))  # (N, C, W, H, D)
                image3D = image3D.unsqueeze(dim=1)

                image3D = image3D.to(device)
                gt3D = gt3D.to(device).float()


                image_embedding = sam_model.image_encoder(image3D)

                self.click_points = []
                self.click_labels = []

                pred_list = []

                prev_masks, loss = self.interaction(sam_model,
                                                    image_embedding,
                                                    gt3D,
                                                    num_clicks=11)

                epoch_loss += loss.item()
                epoch_dice += self.get_dice_score(prev_masks, gt3D)
                cur_loss = loss.item()

                loss /= self.args.accumulation_steps

                self.scaler.scale(loss).backward()

            if step % self.args.accumulation_steps == 0 and step != 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                print_loss = step_loss / self.args.accumulation_steps
                step_loss = 0
                print_dice = self.get_dice_score(prev_masks, gt3D)
            else:
                step_loss += cur_loss

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                if step % self.args.accumulation_steps == 0 and step != 0:
                    print(f'Epoch: {epoch}, Step: {step}, Loss: {print_loss}, Dice: {print_dice}')
                    if print_dice > self.step_best_dice:
                        self.step_best_dice = print_dice
                        if print_dice > 0.9:
                            self.save_checkpoint(epoch,
                                                 sam_model.state_dict(),
                                                 describe=f'{epoch}_step_dice:{print_dice}_best')
                    if print_loss < self.step_best_loss:
                        self.step_best_loss = print_loss

        epoch_loss /= step + 1
        epoch_dice /= step + 1

        return epoch_loss, epoch_iou, epoch_dice, pred_list

    def eval_epoch(self, epoch, num_clicks):
        self.model.eval()
        if self.args.multi_gpu:
            sam_model = self.model.module
        else:
            sam_model = self.model

        val_dataloader = self.val_dataloader

        total_loss = 0
        total_dice = 0
        with torch.no_grad():
            for data3D in tqdm(val_dataloader, desc=f'Validation Epoch {epoch}'):
                image3D, gt3D = data3D["image"], data3D["label"]

                image3D = self.norm_transform(image3D.squeeze(dim=1)).unsqueeze(1).to(device)
                gt3D = gt3D.to(device).float()

                image_embedding = sam_model.image_encoder(image3D)

                self.click_points = []
                self.click_labels = []

                pred_masks, loss = self.interaction(sam_model,
                                                    image_embedding,
                                                    gt3D,
                                                    num_clicks=num_clicks)

                total_loss += loss.item()
                total_dice += self.get_dice_score(pred_masks, gt3D)

        avg_loss = total_loss / len(val_dataloader)
        avg_dice = total_dice / len(val_dataloader)

        return avg_loss, avg_dice

# IN THE plot_result FUNCTION DEFINITION
    def plot_result(self, train_data, val_data=None, description='', save_name='loss'):
        plt.figure()
        plt.plot(train_data, label='Train')
        if val_data is not None:
            plt.plot(val_data, label='Validation')
        plt.title(description)
        plt.xlabel('Epoch')
        plt.ylabel(save_name.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(join(MODEL_SAVE_PATH, f'{save_name}.png'))
        plt.close()


    def train(self):
        self.scaler = torch.amp.GradScaler("cuda")
        for epoch in range(self.start_epoch, self.args.num_epochs):
            print(f'Epoch: {epoch}/{self.args.num_epochs - 1}')

            if self.args.multi_gpu:
                dist.barrier()
                self.dataloaders.sampler.set_epoch(epoch)
            num_clicks = np.random.randint(1, 21)
            epoch_loss, epoch_iou, epoch_dice, pred_list = self.train_epoch(epoch, num_clicks)
            val_loss, val_dice = self.eval_epoch(epoch, num_clicks)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.args.multi_gpu:
                dist.barrier()

            if not self.args.multi_gpu or (self.args.multi_gpu and self.args.rank == 0):
                self.losses.append(epoch_loss)
                self.dices.append(epoch_dice)
                self.val_losses.append(val_loss)
                self.val_dices.append(val_dice)
                print(f'EPOCH: {epoch}, Loss: {epoch_loss}')
                print(f'EPOCH: {epoch}, Dice: {epoch_dice}')
                print(f'Validation Loss: {val_loss}, Validation Dice: {val_dice}')

                logger.info(f'Epoch\t {epoch}\t : loss: {epoch_loss}, dice: {epoch_dice}')
                logger.info(f'Validation - Epoch {epoch} - Loss: {val_loss}, Dice: {val_dice}')


                if self.args.multi_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                # save latest checkpoint
                self.save_checkpoint(epoch, state_dict, describe='latest')

                # save train loss best checkpoint
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(epoch, state_dict, describe='val_loss_best')


                # save train dice best checkpoint
                if val_dice > self.best_dice:
                    self.best_dice = val_dice
                    self.save_checkpoint(epoch, state_dict, describe='val_dice_best')


                self.plot_result(self.losses, self.val_losses, description='Dice + Cross Entropy Loss', save_name='Loss')
                self.plot_result(self.dices, self.val_dices, description='Dice', save_name='Dice')
        logger.info('=====================================================================')
        logger.info(f'Best loss: {self.best_loss}')
        logger.info(f'Best dice: {self.best_dice}')
        logger.info(f'Total loss: {self.losses}')
        logger.info(f'Total dice: {self.dices}')
        logger.info('=====================================================================')
        logger.info(f'args : {self.args}')
        logger.info(f'Used datasets : {img_datas}')
        logger.info('=====================================================================')


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def device_config(args):
    try:
        if not args.multi_gpu:
            # Single GPU
            if args.device == 'mps':
                args.device = torch.device('mps')
            else:
                args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
        else:
            args.nodes = 1
            args.ngpus_per_node = len(args.gpu_ids)
            args.world_size = args.nodes * args.ngpus_per_node

    except RuntimeError as e:
        print(e)




def main():
    mp.set_sharing_strategy('file_system')
    device_config(args)
    if args.multi_gpu:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args, ))
    else:
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        # Load datasets
        train_dataloader, val_dataloader = get_dataloaders(args)

        # Build model
        model = build_model(args)

        # Create trainer
        trainer = BaseTrainer(model, (train_dataloader, val_dataloader), args)

        # Train
        trainer.train()


def main_worker(rank, args):
    setup(rank, args.world_size)

    torch.cuda.set_device(rank)
    args.num_workers = int(args.num_workers / args.ngpus_per_node)
    args.device = torch.device(f"cuda:{rank}")
    args.rank = rank

    init_seeds(2023 + rank)

    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO if rank in [-1, 0] else logging.WARN,
                        filemode='w',
                        filename=os.path.join(LOG_OUT_DIR, f'output_{cur_time}.log'))

    dataloaders = get_dataloaders(args)
    model = build_model(args)
    trainer = BaseTrainer(model, dataloaders, args)
    trainer.train()
    cleanup()


def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group(backend='nccl',
                            init_method=f'tcp://127.0.0.1:{args.port}',
                            world_size=world_size,
                            rank=rank)


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

