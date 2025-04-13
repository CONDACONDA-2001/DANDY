import math
import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from net.TJNet import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import optim, nn
import configparser
from torch.nn.parallel import DataParallel


def calculate_metrics(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()

    # MAE
    mae = np.mean(np.abs(pred - gt))

    # 二值化计算其他指标
    th = 0.5
    bin_pred = (pred > th).astype(np.float32)

    # FM
    tp = np.sum(bin_pred * gt)
    precision = tp / (np.sum(bin_pred) + 1e-8)
    recall = tp / (np.sum(gt) + 1e-8)
    fmeasure = (1.3 * precision * recall) / (0.3 * precision + recall + 1e-8)

    # SM
    y_mean = np.mean(gt)
    if y_mean == 0:
        sm = 1 - np.mean(pred)
    elif y_mean == 1:
        sm = np.mean(pred)
    else:
        sm = np.abs(np.mean(pred) - y_mean) * 2

    # EM
    align_matrix = 2 * gt - 1
    em = np.mean(align_matrix * (2 * pred - 1))

    # wFM
    weighted_fm = fmeasure * (1 - np.abs(np.mean(pred) - np.mean(gt)))

    return mae, fmeasure, sm, em, weighted_fm


def structure_loss(pred, mask):
    '''
    通过加权交叉熵和 IoU 损失增强边界学习
    '''
    weit = 1 + 5 * \
           torch.abs(F.avg_pool2d(mask, kernel_size=31,
                                  stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        '''
        Focal loss:动态降低易分类样本的损失权重，迫使模型更关注边缘信息
        Dice loss:优化 IoU 指标（与真实值重合度）,对类别不平衡不敏感
        '''
        # Focal Loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce

        # Dice Loss
        smooth = 1
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice = (2. * intersection + smooth) / (pred_sigmoid.sum() + target.sum() + smooth)

        return 0.5 * focal_loss.mean() + 0.5 * (1 - dice)


def get_adaptive_weight(pred, target):
    '''
    动态调整各层损失权重，技术实现要点：
    1. 通过Sigmoid获取概率图(pred_p)
    2. 计算预测与GT的交并比：2*|pred ∩ gt| / (|pred| + |gt|)
    3. 权重公式：1 - 交并比 → 预测与真值差异越大，权重越高
    '''
    with torch.no_grad():
        pred_p = torch.sigmoid(pred)
        intersection = (pred_p * target).sum()
        union = pred_p.sum() + target.sum()
        return 1 - 2 * intersection / (union + 1e-8)


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) *
                    valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p))
                    * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    model.train()
    total_loss = 0
    for i, (images, gts, edges) in enumerate(train_loader, 1):
        images, gts, edges = images.cuda(), gts.cuda(), edges.cuda()

        lateral_map_3, lateral_map_2, lateral_map_1, edge_map, coarse_map = model(images)

        # 自适应权重,提升鲁棒性
        w3 = get_adaptive_weight(lateral_map_3, gts)
        w2 = get_adaptive_weight(lateral_map_2, gts)
        w1 = get_adaptive_weight(lateral_map_1, gts)

        loss_fn = HybridLoss()
        loss3 = w3 * loss_fn(lateral_map_3, gts)
        loss2 = w2 * loss_fn(lateral_map_2, gts)
        loss1 = w1 * loss_fn(lateral_map_1, gts)
        losse = 2.0 * dice_loss(edge_map, edges)
        lossc = 0.5 * structure_loss(coarse_map, gts)

        loss = loss3 + loss2 + loss1 + losse + lossc

        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        total_loss += loss.item()

        if i % 20 == 0 or i == total_step or i == 1:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}], [coarse: {:,.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss3.data, loss2.data, loss1.data, losse.data, lossc.data))
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                         '[lateral-3: {:.4f}], [lateral-2: {:.4f}], [lateral-1: {:.4f}], [edge: {:,.4f}], [coarse: {:,.4f}]'.
                         format(datetime.now(), epoch, opt.epoch, i, total_step,
                                loss3.data, loss2.data, loss1.data, losse.data, lossc.data))
            # TensorboardX-Loss
            writer.add_scalars('Loss_Statistics',
                               {'Loss1': loss1.data, 'Lossc': lossc.data,
                                'Losse': losse.data, 'Loss_total': loss.data},
                               global_step=step)
            # TensorboardX-Training Data
            grid_image = make_grid(
                images[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('RGB', grid_image, step)
            grid_image = make_grid(
                gts[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('GT', grid_image, step)
            grid_image = make_grid(
                edges[0].clone().cpu().data, 1, normalize=True)
            writer.add_image('Edge', grid_image, step)

            # TensorboardX-Outputs
            res = coarse_map[0].clone()
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            writer.add_image('Pred_coarse', torch.tensor(
                res), step, dataformats='HW')

            res = lateral_map_1[0].clone()
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            writer.add_image('Pred_final', torch.tensor(
                res), step, dataformats='HW')

            res = edge_map[0].clone()
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            writer.add_image('Pred_edge', torch.tensor(
                res), step, dataformats='HW')

            if epoch % 80 == 0:
                torch.save(model.module.state_dict(), save_path +
                           'TJNet_{}.pth'.format(epoch))



best_metrics = {
    'MAE': [1.0, 1.0, 1.0],
    'FM': [0.0, 0.0, 0.0],
    'SM': [0.0, 0.0, 0.0],
    'EM': [0.0, 0.0, 0.0],
    'wFM': [0.0, 0.0, 0.0]
}
best_epoch = [0, 0, 0]


def val(test_loader, model, epoch, save_path, writer, val_dataset, val_idx):
    global best_metrics, best_epoch
    model.eval()
    metrics = {'MAE': 0, 'FM': 0, 'SM': 0, 'EM': 0, 'wFM': 0}
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32) / 255

            # 转换GT为PyTorch张量并调整维度
            gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).float().cuda()

            image = image.cuda()

            _, _, res, _, _ = model(image)

            # 调整res维度
            if res.dim() == 2:
                res = res.unsqueeze(0).unsqueeze(0)
            elif res.dim() == 3:
                res = res.unsqueeze(1)

            # 插值到GT尺寸
            res = F.interpolate(res, size=gt.shape, mode='bilinear')  # 直接使用gt的原始尺寸

            res = res.sigmoid().cpu().numpy().squeeze()

            mae, fm, sm, em, wfm = calculate_metrics(res, gt.squeeze())
            metrics['MAE'] += mae
            metrics['FM'] += fm
            metrics['SM'] += sm
            metrics['EM'] += em
            metrics['wFM'] += wfm

        for k in metrics:
            metrics[k] /= test_loader.size
            writer.add_scalar(f'{k}_{val_dataset}', metrics[k], epoch)

        print(f'Epoch: {epoch} | MAE: {metrics["MAE"]:.4f} | FM: {metrics["FM"]:.4f} | '
              f'SM: {metrics["SM"]:.4f} | EM: {metrics["EM"]:.4f} | wFM: {metrics["wFM"]:.4f}')
        # 更新全局最佳指标
        if metrics['MAE'] < best_metrics['MAE'][val_idx] and metrics['FM'] > best_metrics['FM'][val_idx]:
            for k in metrics:
                best_metrics[k][val_idx] = metrics[k]
            best_epoch[val_idx] = epoch
            torch.save(model.module.state_dict(), save_path + 'TJNet_best_{}.pth'.format(val_dataset))
            print('Save state_dict successfully! Best epoch:{}, val_dataset:{}.'.format(epoch, val_dataset))
        logging.info(
            '[Val Info]:val_dataset:{} Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(val_dataset, epoch, mae,
                                                                                       best_epoch[val_idx],
                                                                                       best_mae[val_idx]))
    # 多指标判断模型保存
    if metrics['MAE'] < best_metrics['MAE'][val_idx] and metrics['FM'] > best_metrics['FM'][val_idx]:
        torch.save(model.module.state_dict(), f'{save_path}TJNet_best_MilitaryCamouflage.pth')
        for k in metrics:
            best_metrics[k][val_idx] = metrics[k]


if __name__ == '__main__':
    import argparse

    config = configparser.ConfigParser(allow_no_value=True)
    config.read('./config.ini')
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=config['Train'].getint('epoch'), help='epoch number')
    parser.add_argument('--lr', type=float, default=config['Train'].getfloat('lr'), help='learning rate')
    parser.add_argument('--batchsize', type=int, default=config['Train'].getint('batchsize'),
                        help='training batch size')
    parser.add_argument('--trainsize', type=int, default=config['Comm'].getint('trainsize'),
                        help='training dataset size')
    parser.add_argument('--clip', type=float, default=config['Train'].getfloat('clip'),
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=config['Train'].getfloat('decay_rate'), help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=config['Train'].getint('decay_epoch'),
                        help='every n epochs decay learning rate')
    __load__ = config['Train']['load']
    if __load__.lower() == 'none':
        __load__ = None
    parser.add_argument('--load', type=str, default=__load__,
                        help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default=config['Train']['train_root'],
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default=config['Train']['val_root'],
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str, default=config['Train']['save_path'],
                        help='the path to save model and log')
    parser.add_argument('--net_channel', type=int, default=config['Comm'].getint('net_channel'))
    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    cudnn.benchmark = True


    model = Network(opt.net_channel)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    # 优化后的学习率策略，采用余弦退火机制平滑降低学习率
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (opt.epoch - warmup_epochs)))


    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    val_loader_MilitaryCamouflage = test_dataset(image_root=opt.val_root + 'MilitaryCamouflage/Imgs/',
                                                 gt_root=opt.val_root + 'MilitaryCamouflage/GT/',
                                                 testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_mae = [1, 1, 1]
    best_epoch = [0, 0, 0]

    # 调整学习率（线性缩放规则）
    base_lr = opt.lr * torch.cuda.device_count()  # 学习率按GPU数量缩放
    cosine_schedule = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print("Start train...")

    for epoch in range(1, opt.epoch):

        # 记录学习率
        current_lr = cosine_schedule.get_last_lr()[0]
        writer.add_scalar('learning_rate', current_lr, epoch)
        logging.info(f'Current lr: {current_lr}')

        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader_MilitaryCamouflage, model, epoch, save_path, writer, "MilitaryCamouflage", 0)
