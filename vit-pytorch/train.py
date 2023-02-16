import time
from collections import OrderedDict
from options.trainoptions import TrainOptions
from libs.datasets import data_loader
from libs.general import util as util
import os
import numpy as np
from network import *


from torch.autograd import Variable
import time


opt=TrainOptions().parse()

iter_path=os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
iou_path=os.path.join(opt.checkpoints_dir, opt.name, 'iou.txt')

if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    try:
        best_iou = np.loadtxt(iou_path, delimiter=',', dtype=float)

    except:
        best_iou = 0

    print('Resuming from epoch %d at iteration %d, best iou is %.3f' % (start_epoch, epoch_iter, best_iou))
else:
    start_epoch, epoch_iter, best_iou = 1, 0, 0


data_loader=data_loader.Create_DataLoader(opt)
dataset, dataset_val= data_loader.CustomDataLoader.load_data(data_loader)
dataset_size=len(dataset)

print('#training images = %d' % dataset_size)

model=ViT(opt)

model.cuda()

total_steps=(start_epoch-1)*dataset_size+epoch_iter

for epoch in range(start_epoch, opt.niter+opt.niter_decay+1):
    epoch_start_time= time.time()
    if epoch != start_epoch:
        epoch_iter=epoch_iter%dataset_size
    model.model.train()
    for i, data in enumerate (dataset, start=epoch_iter):
        iter_start_time= time.time()
        total_steps+=opt.batchSize
        epoch_iter += opt.batchSize

        ''' forward pass'''

        model.forward(data)

        model.backward(total_steps, opt.nepochs*dataset.__len__()*opt.batchSize+1)


        ''' Save latest model'''

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')
    model.model.eval()
    if dataset_val!=None:
        label_trues, label_preds = [], []
        for i, data in enumerate (dataset_val):
            segt, segpred = model.forward(data, False)
            segt=segt.data.cpu().numpy()
            segpred=segpred.data.cpu().numpy()

            label_trues.append(segt)
            label_preds.append(segpred)

        metrics =util.label_accuracy_score(label_trues, label_preds, n_class=opt.label_nc)
        metrics = np.array(metrics)
        metrics*=100

        print('''\
            Validation:
            Accuracy: {0}
            Accuracy Class: {1}
            Mean IU: {2}
            FWAV Accuracy: {3}'''.format(*metrics)
            )
        
        if metrics[2]>best_iou:
            best_iou=metrics[2]
            model.save('best')
            np.savetxt(iou_path, (best_iou), delimiter=',', fmt='%f')



#write a pytorch training loop for a vision transformer model

