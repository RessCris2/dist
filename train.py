"""
  训练 UNetDIST 模型
"""
import time
import torch
from torch import nn
from dist_net import DIST, loss_fn
from dataloader import DISTDataset
from torch.utils.data import DataLoader
from  os.path import join as opj
import sys
sys.path.append("/root/autodl-tmp/archive")
from  metrics.compute_inst import run_one_inst_stat
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from predict import predict
import logging
from metrics.utils import get_logger, rm_n_mkdir, load_model, load_img, find_files, get_curtime
############## TENSORBOARD ########################
import sys
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
# writer = SummaryWriter('/root/autodl-tmp/archive/v2/model_data/dist/pannuke/tensor_log')


###################################################

def evaluate_one_pic(model ):
    """只针对一张图片计算指标结果
    """
    path = "/root/autodl-tmp/datasets/pannuke/inst/test/0.npy"
    true = np.load(path)
    # 只对一张图片进行评估
    img = load_img("/root/autodl-tmp/datasets/pannuke/coco_format/images/test/0.jpg")
    pred = predict(model, img)
    metrics = run_one_inst_stat(true, pred, match_iou=0.5)
    return pred, metrics


def train(dataset_name, model_name='dist'):
    """
        
    """
    epochs = 50
    batch_size = 8
    val_interval = 2
    save_interval = 5
    num_features = 6  # 这个参数可以调节模型的复杂度
    dir_root = "/root/autodl-tmp/archive/datasets/{}/patched/coco_format/".format(dataset_name)
    save_dir  = "/root/autodl-tmp/archive/v2/model_data/{}/{}/{}".format(model_name, dataset_name, get_curtime())
    rm_n_mkdir(save_dir)
    logger = get_logger(log_file_name='train.log', log_dir='/root/autodl-tmp/archive/v2/model_data/dist/{}/'.format(dataset_name))

    writer_save_dir = '/root/tf-logs/{}/{}/{}'.format(dataset_name, model_name, get_curtime)
    rm_n_mkdir(writer_save_dir)
    writer = SummaryWriter(writer_save_dir)

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = dict()
    classes.update({'pannuke':['Background','Neoplastic','Inflammatory','Connective','Dead','Epithelial']})
    classes.update({'monusac': ['Background','Epithelial','Lymphocyte','Neutrophil','Macrophage']})
    classes.update({'consep': ['background','inflammatory','healthy_epithelial','epithelial', 'spindle-shaped']})
    classes.update({'cpm17':None}) # 处理一下分类数据为 None 的情况，应该是读入的时候就不一样

    # 训练集
    train_images_dir = opj(dir_root, 'images/train')
    train_masks_dir =  opj(dir_root, 'seg_mask/train')
    train_loader = DataLoader(dataset=DISTDataset(train_images_dir, train_masks_dir, classes=classes[dataset_name]),
                              batch_size = batch_size,
                              shuffle=True,
                              drop_last=False,
                              num_workers=8)
    # 测试机
    # dir_root = "/root/autodl-tmp/datasets/pannuke/coco_format/"
    test_images_dir = opj(dir_root, 'images/test')
    test_masks_dir =  opj(dir_root, 'seg_mask/test')
    test_loader = DataLoader(dataset=DISTDataset(test_images_dir, test_masks_dir, classes=classes[dataset_name]),
                              batch_size = batch_size,
                              shuffle=False,
                              drop_last=False,
                              num_workers=8)


    # examples = iter(test_loader)
    # example_data, example_targets = next(examples)
    # for i in range(2):
    #     plt.subplot(2,3,i+1)
    #     plt.imshow(example_data[i][0], cmap='gray')
    # #plt.show()
    example_data = []
    for iter, (image, label) in enumerate(test_loader):
        if iter < 6:
            image, label = image.to(device), label.to(device)
            example_data.append(image[0])
        else:
            break

    ############## TENSORBOARD ########################
    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image('test_images', img_grid)
    #writer.close()
    #sys.exit()
    ###################################################

    model = DIST(num_features=num_features).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.01)
    
    ############## TENSORBOARD ########################
    # writer.add_graph(model, example_data.to(device))
    #writer.close()
    #sys.exit()
    ###################################################

    n_total_steps = len(train_loader)
    for epoch in tqdm(range(epochs)):
        start = time.perf_counter() 
        model = model.cuda()
        for iter_idx, (image, label) in enumerate(train_loader):
            model.train()
            image, label = image.to(device), label.to(device)
            pred = model(image) 
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            # 假如每个 iter 都要保存数据
            ############## TENSORBOARD ########################
            writer.add_scalar('training loss', loss, epoch * n_total_steps + iter_idx)
            # running_accuracy = running_correct / 100 / predicted.size(0)
            # writer.add_scalar('accuracy', running_accuracy, epoch * n_total_steps + i)
            # running_correct = 0
            # running_loss = 0.0
            ###################################################

        # 每隔保存 interval 保存模型
        if epoch % save_interval == 0:
            save_path = opj(save_dir, "epoch_{}.pth".format(epoch))
            torch.save(model.state_dict(), save_path)

        # 每隔保存 interval 保存模型
        n_test_steps = len(test_loader)
        if epoch % val_interval == 0:
            for iter_idx, (image, label) in enumerate(test_loader):
                model.eval()
                image, label = image.to(device), label.to(device)
                pred = model(image) 
                loss = loss_fn(pred, label)
                writer.add_scalar('validataion loss', loss, epoch * n_test_steps + iter_idx)
        dur = time.perf_counter() - start    # 计时，计算进度条走到某一百分比的用时
        logger.info("epoch_{} 耗费时间为{}s".format(epoch, dur))

if __name__ == "__main__":
    dataset_name='consep'
    model_name = 'dist'
    train(dataset_name, model_name)
    # model = DIST(num_features=6)
    # model = load_model(model, path = "/root/autodl-tmp/com_models/dist_torch/model_data/epoch_10.pth")
    # pred, _ = evaluate_one_pic(model)
