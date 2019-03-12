import time,os
import torch
import shutil
import argparse
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from layers.modules import MultiBoxLoss
from data import mk_anchors
from data import COCODetection, VOCDetection, detection_collate, preproc
from configs.CC import Config
from termcolor import cprint
from utils.nms_wrapper import nms
import numpy as np

def set_logger(status):
    if status:
        from logger import Logger
        date = time.strftime("%m_%d_%H_%M") + '_log'
        log_path = './logs/'+ date
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)
        logger = Logger(log_path)
        return logger
    else:
        pass

def anchors(cfg):
    return mk_anchors(cfg.model.input_size,
                               cfg.model.input_size,
                               cfg.model.anchor_config.size_pattern, 
                               cfg.model.anchor_config.step_pattern)
    
def init_net(net, cfg, resume_net):    
    if cfg.model.init_net and not resume_net:
        net.init_model(cfg.model.pretrained)
    else:
        print('Loading resume network...')
        state_dict = torch.load(resume_net)

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict,strict=False)

def set_optimizer(net, cfg):
    return optim.SGD(net.parameters(),
                     lr = cfg.train_cfg.lr[0],
                     momentum = cfg.optimizer.momentum,
                     weight_decay = cfg.optimizer.weight_decay)

def set_criterion(cfg):
    return MultiBoxLoss(cfg.model.m2det_config.num_classes,
                        overlap_thresh = cfg.loss.overlap_thresh,
                        prior_for_matching = cfg.loss.prior_for_matching,
                        bkg_label = cfg.loss.bkg_label,
                        neg_mining = cfg.loss.neg_mining,
                        neg_pos = cfg.loss.neg_pos,
                        neg_overlap = cfg.loss.neg_overlap,
                        encode_target = cfg.loss.encode_target)

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size, cfg):
    global lr
    if epoch <= 5:
        lr = cfg.train_cfg.end_lr + (cfg.train_cfg.lr[0]-cfg.train_cfg.end_lr)\
         * iteration / (epoch_size * cfg.train_cfg.warmup)
    else:
        for i in range(len(cfg.train_cfg.step_lr.COCO)):
            if cfg.train_cfg.step_lr.COCO[i]>=epoch:
                lr = cfg.train_cfg.lr[i]
                break
        # lr = cfg.train_cfg.init_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_dataloader(cfg, dataset, setname='train_sets'):
    _preproc = preproc(cfg.model.input_size, cfg.model.rgb_means, cfg.model.p)
    Dataloader_function = {'VOC': VOCDetection, 'COCO':COCODetection}
    _Dataloader_function = Dataloader_function[dataset]
    if setname == 'train_sets':
        dataset = _Dataloader_function(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot,
                                   getattr(cfg.dataset, dataset)[setname], _preproc)
    else:
        dataset = _Dataloader_function(cfg.COCOroot if dataset == 'COCO' else cfg.VOCroot,
                                   getattr(cfg.dataset, dataset)[setname], None)
    return dataset
    
def print_train_log(iteration, print_epochs, info_list):
    if iteration % print_epochs == 0:
        cprint('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss_L:{:.4f}||Loss_C:{:.4f}||Batch_Time:{:.4f}||LR:{:.7f}'.format(*info_list), 'green')
       
def print_info(info, _type=None):
    if _type is not None:
        if isinstance(info,str):
            cprint(info, _type[0], attrs=[_type[1]])
        elif isinstance(info,list):
            for i in range(info):
                cprint(i, _type[0], attrs=[_type[1]])
    else:
        print(info)


def save_checkpoint(net, cfg, final=True, datasetname='COCO',epoch=10):
    if final:
        torch.save(net.state_dict(), cfg.model.weights_save + \
                'Final_M2Det_{}_size{}_net{}.pth'.format(datasetname, cfg.model.input_size, cfg.model.m2det_config.backbone))
    else:
        torch.save(net.state_dict(), cfg.model.weights_save + \
                'M2Det_{}_size{}_net{}_epoch{}.pth'.format(datasetname, cfg.model.input_size, cfg.model.m2det_config.backbone,epoch))



def write_logger(info_dict,logger,iteration,status):
    if status:
        for tag,value in info_dict.items():
            logger.scalar_summary(tag, value, iteration)
    else:
        pass

def image_forward(img, net, cuda, priors, detector, transform):
    w,h = img.shape[1],img.shape[0]
    scale = torch.Tensor([w,h,w,h])
    with torch.no_grad():
        x = transform(img).unsqueeze(0)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()
    out = net(x)
    boxes, scores = detector.forward(out, priors)
    boxes = (boxes[0] * scale).cpu().numpy()
    scores = scores[0].cpu().numpy()
    return boxes, scores
   
def nms_process(num_classes, i, scores, boxes, cfg, min_thresh, all_boxes, max_per_image):
    for j in range(1, num_classes): # ignore the bg(category_id=0)
        inds = np.where(scores[:,j] > min_thresh)[0]
        if len(inds) == 0:
            all_boxes[j][i] = np.empty([0,5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)

        soft_nms = cfg.test_cfg.soft_nms
        keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
        keep = keep[:cfg.test_cfg.keep_per_class] # keep only the highest boxes
        c_dets = c_dets[keep, :]
        all_boxes[j][i] = c_dets
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]


