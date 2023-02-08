'''
New Dataloader for NYU RGB-D Dataset
Scripts loads and creates a custom dataset

Author: Mitterand Ekole
'''

import torch.utils.data

class BaseDataLoader():
    def __init__(self):
        pass
    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None

 #create dataset using NYU RGB-D dataset   
def CreateDataset(opt):
    dateset=None
    if opt.dataset_mode == 'nyu':
        from datasets.nyu_dataset_crop import NYUDataset, NYUDataset_val
        dataset = NYUDataset()
        if opt.vallist!='':
            dataset_val = NYUDataset_val()
        else:
            dataset_val = None
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)
    
    return dataset, dataset_val

class CustomDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDataLoader'
    
    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset, self.dataset_val = CreateDataset(opt)
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        if self.dataset_val !=None:
            self.data_loader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=1,
                shuffle=False,
                num_workers=int(opt.nThreads))
        else:
            self.data_loader_val = None
    
    def load_data(self):
        return self.data_loader, self.data_loader_val
    
    def len(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    

def CreateDataLoader(opt):
    data_loader = CustomDataLoader()
    data_loader.initialize(opt)
    return data_loader
    
