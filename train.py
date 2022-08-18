import os
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, nn
from dataset_loader import MyDataset
from models.bicnn_att_model import CNNTransformer
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
from losses.loss import BinaryDiceLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)


def mkdirs(path):
    if not os.path.exists(path): os.makedirs(path)


def dice_metric(output, target):
    output = output > 0
    dice = ((output * target).sum() * 2+0.1) / (output.sum() + target.sum() + 0.1)
    return dice

def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    #model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model

def voe_metric(output, target):
    output = output > 0
    voe = ((output.sum() + target.sum() - (target*output).sum().float()* 2) +0.1) / (output.sum() + target.sum() - (target*output).sum().float()+ 0.1)
    return voe.item()

def rvd_metric(output, target):
    output = output > 0
    rvd = ((output.sum() / (target.sum() + 0.1) - 1) * 100)
    return rvd.item()


def train_epoch(epoch, model, dl, optimizer, criterion, criterion2):
    model.train()
    bar = tqdm(dl)
    bar.set_description_str("%02d" % epoch)
    loss_v, dice_v, ii = 0, 0, 0
    for x2, mask in bar:
        x2 = rearrange(x2, 'b 1 (n h) (m w) -> (b n m) 1 h w', n=4,m=4,h=64,w=64)
        outputs = model(x2.float().to(device))
        mask = rearrange(mask, 'b 1 (n h) (m w) -> (b n m) 1 h w', n=4,m=4,h=64,w=64)
        mask = mask.float().to(device)
        loss2 = criterion2(outputs,mask)
        loss1 = criterion(outputs, mask)
                
        loss = loss1+0.01*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dice = dice_metric(outputs, mask)
        dice_v += dice
        loss_v += loss.item()
        ii += 1
        bar.set_postfix(loss=loss.item(),loss1=loss1.item(),loss2=loss2.item(), dice=dice)
    return loss_v / ii, dice_v / ii


@torch.no_grad()
def val_epoch(model, dl, criterion):
    model.eval()
    loss_v, dice_v, voe_v, rvd_v, ii = 0, 0, 0, 0, 0
    for x2, mask in dl:
        x2 = rearrange(x2, 'b 1 (n h) (m w) -> (b n m) 1 h w', n=4,m=4,h=64,w=64)
        outputs = model(x2.float().to(device))
        mask = rearrange(mask, 'b 1 (n h) (m w) -> (b n m) 1 h w', n=4,m=4,h=64,w=64)
        mask = mask.float().to(device)
        loss_v += criterion(outputs, mask).item()
        dice_v += dice_metric(outputs, mask)
        voe_v += voe_metric(outputs, mask)
        rvd_v += rvd_metric(outputs, mask)
        ii += 1
    return loss_v / ii, dice_v / ii, voe_v / ii, rvd_v / ii


def train(opt):
    model = CNNTransformer(in_channels=1)
    if opt.w:
        model.load_state_dict(torch.load(opt.w))
    model = model.to(device)
    model = nn.DataParallel(model)

    root_dir = '../dataset/3Dircadb_2d_dataset'
    #root_dir = '../dataset/MSD_2d_dataset'
    train_image_root = 'train'
    val_image_root = 'val'

    train_dataset = MyDataset(model_type=train_image_root, data_filename=root_dir,sub_name='',transform=transforms.ToTensor())
    train_dl = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True)
    val_dataset = MyDataset(model_type=val_image_root, data_filename=root_dir,sub_name='',transform=transforms.ToTensor())
    val_dl = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=4, shuffle=False)

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    criterion2 = BinaryDiceLoss()
    # 一些文件日志信息
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6,factor=0.1,patience=5)
    best_dice_epoch, best_dice, b_voe, b_rvd, train_loss, train_dice = 0, 0, 0, 0, 0, 0
    pre_loss,sur_loss = 0,0
    save_dir = os.path.join(opt.ckpt, datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + "_" + opt.name
    mkdirs(save_dir)
    w_dice_best = os.path.join(save_dir, '3dir_dice_best_1.pth')

    fout_log = open(os.path.join(save_dir, '3dir_log_1.txt'), 'w')
    print(len(train_dataset), len(val_dataset), save_dir)
    for epoch in range(opt.max_epoch):
        if not opt.eval:
            train_loss, train_dice = train_epoch(epoch, model, train_dl, optimizer, criterion, criterion2)
        val_loss, val_dice, voe_v, rvd_v = val_epoch(model, val_dl, criterion)
        if best_dice < val_dice:
            best_dice, best_dice_epoch, b_voe, b_rvd = val_dice, epoch, voe_v, rvd_v
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), w_dice_best)

        lr = optimizer.param_groups[0]['lr']
        log = "%02d train_loss:%0.3e, train_dice:%0.5f, val_loss:%0.3e, val_dice:%0.5f, lr:%.3e\n best_dice:%.5f, voe:%.5f, rvd:%.5f(%02d)\n" % (
            epoch, train_loss, train_dice, val_loss, val_dice, lr, best_dice, b_voe, b_rvd, best_dice_epoch)
        print(log)
        fout_log.write(log)
        fout_log.flush()
        scheduler.step(val_loss)
        #cur = cur + 1
    fout_log.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='setr', help='study name')
    parser.add_argument('--batch_size', type=int, default=6, help='batch size')
    parser.add_argument('--input_size', type=int, default=512, help='input size')
    parser.add_argument('--max_epoch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='ckpt', help='the dir path to save model weight')
    parser.add_argument('--w', type=str, help='the path of model wight to test or reload')
    parser.add_argument('--suf', type=str, choices=['.dcm', '.JL', '.png'], help='suffix', default='.png')
    parser.add_argument('--eval', action="store_true", help='eval only need weight')
    parser.add_argument('--test_root', type=str, help='root_dir')

    opt = parser.parse_args()
    train(opt)
