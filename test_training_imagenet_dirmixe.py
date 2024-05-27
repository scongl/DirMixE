import argparse
import os
import torch
from tqdm import tqdm
from pathlib import Path
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import numpy as np
from parse_config import ConfigParser
import torch.nn.functional as F
import torch
import random
import numpy as np
import os, sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Sampler
from base import BaseDataLoader
from PIL import Image
from PIL import ImageFilter
import torch.backends.cudnn as cudnn


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeters(object):
    def __init__(self, size):
        self.meters = [AverageMeter(i) for i in range(size)]
    
    def update(self, idxs, vals):
        for i, v in zip(idxs, vals):
            self.meters[i].update(v)
    
    def get_avgs(self):
        return np.array([m.avg for m in self.meters])
    
    def get_sums(self):
        return np.array([m.sum for m in self.meters])
    
    def get_cnts(self):
        return np.array([m.count for m in self.meters])


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
    
class BalancedSampler(Sampler):
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])  
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num 

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels # Sampler needs to use targets
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class DirichletImbalanceImageNet(Dataset):
    def __init__(self, root, train_cls_num_list, imb_type='exp', imb_factor=0.01, train=True,
                 transform=None, target_transform=None, download=False, reverse=False,
                 txt='data_txt/ImageNet_LT/ImageNet_LT_uniform.txt'):
        
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels # Sampler needs to use targets
        
        cls_num = 1000
        
        img_num_per_cls = self.get_img_per_cls(cls_num, imb_type, imb_factor, reverse, train_cls_num_list)
        self.gen_imbalanced_data(img_num_per_cls)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        # return sample, label, path
        return sample, label
        
    def get_img_per_cls(self, cls_num, imb_type, imb_factor, reverse, train_cls_num_list):
        img_max = len(self.img_path) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num =  img_max * (imb_factor**((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))                    
                else:
                    num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        
        sorted_idx = np.argsort(train_cls_num_list)
        sorted_idx = sorted_idx[::-1]
        
        cls_num_list = [0] * cls_num
        
        for i in range(len(cls_num_list)):
            cls_num_list[sorted_idx[i]] = img_num_per_cls[i]
        
        alpha = cls_num_list
        alpha = [float(x) for x in alpha]
        
        # add 5% pertubation to alpha
        for i in range(len(alpha)):
            alpha[i] += (np.random.random() - 0.5) * 0.1 * alpha[i]
            alpha[i] = max(alpha[i], 1e-7)
            
        alpha_norm = sum(alpha)
        alpha = [x * 10000 / alpha_norm for x in alpha]
        
        img_num_per_cls = np.random.dirichlet(alpha)
        
        pro_max = max(img_num_per_cls)
        img_num_per_cls = [max(int(x / pro_max * img_max), 1) for x in img_num_per_cls]
        
        return img_num_per_cls
        
        
    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.labels, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # new_data.append(self.img_path[selec_idx, ...])
            new_data.extend([self.img_path[i] for i in selec_idx])
            new_targets.extend([the_class, ] * the_img_num)
        # new_data = np.vstack(new_data)
        
        self.img_path = new_data
        self.labels = new_targets
    
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
        

class DirichletImbalanceImageNetDataLoader(DataLoader):
    """
    Imbalance Cifar100 Data Loader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1, training=True, balanced=False, retain_epoch_size=True, imb_type='exp', imb_factor=0.01, test_imb_factor=0, reverse=False, train_txt='./data_txt/ImageNet_LT/ImageNet_LT_train.txt'):
        
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        dataset = LT_Dataset(data_dir, train_txt, train_trsfm)
        num_classes = len(np.unique(dataset.targets))
        assert num_classes == 1000
        
        cls_num_list = [0] * num_classes
        for label in dataset.targets:
            cls_num_list[label] += 1

        self.cls_num_list = cls_num_list
        
        train_dataset = DirichletImbalanceImageNet(data_dir, self.cls_num_list, train=False, download=True, transform= TwoCropsTransform(train_trsfm), imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse) 
        val_dataset = DirichletImbalanceImageNet(data_dir, self.cls_num_list, train=False, download=True, transform=test_trsfm, imb_type=imb_type, imb_factor=test_imb_factor, reverse=reverse) 
            
        self.dataset = dataset
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size, 
            'num_workers': num_workers
        }

        super().__init__(dataset=self.dataset, **self.init_kwargs) # Note that sampler does not apply to validation set

    def train_set(self):
        return DataLoader(dataset=self.train_dataset, shuffle=True, **self.init_kwargs)

    def test_set(self):
        return DataLoader(dataset=self.val_dataset, shuffle=False, **self.init_kwargs)


def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float() \
                       + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def main(config, args):
    args = args.parse_args()
    
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    logger = config.get_logger('test') 
    
    # build model architecture 
    model = config.init_obj('arch', module_arch)
       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    
    # fix numpy seed to get a fixed test dataset
    fix_numpy_seed(args.seed)

    # prepare model for testing
    model = model.to(device)
    weight_record_list=[]
    performance_record_list=[]
    
    distrb = {
        'uniform': (1, False),
        'forward': (args.ir, False),
        'backward': (args.ir, True)
    }
    
    test_distribution_set = ['uniform', 'forward', 'backward']
    
    trials = 9
    idx_list = []
     
    for i in range(trials):
        idx = i // 3
        idx_list.append(idx)
        
        test_distribution = test_distribution_set[idx]
        
        print(test_distribution)
        data_loader = DirichletImbalanceImageNetDataLoader(
            config['data_loader']['args']['data_dir'],
            batch_size=64,
            shuffle=False,
            training=False,
            num_workers=8,
            test_imb_factor=distrb[test_distribution][0],
            reverse=distrb[test_distribution][1]
        )
        
        train_data_loader= data_loader.train_set()
        valid_data_loader = data_loader.test_set()
        num_classes = config._config["arch"]["args"]["num_classes"]        
        aggregation_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
        aggregation_weight.data.fill_(1/3)  
        
        optimizer = config.init_obj('optimizer', torch.optim, [aggregation_weight])
        
        for k in range(config["epochs"]):
            weight_record = test_training(train_data_loader, model, aggregation_weight, optimizer, num_classes, config, args)
            if weight_record[0]<0.05 or weight_record[1]<0.05 or weight_record[2]<0.05:
                break
        print("Aggregation weight: Expert 1 is {0:.2f}, Expert 2 is {1:.2f}, Expert 3 is {2:.2f}".format(weight_record[0], weight_record[1], weight_record[2]))    
        weight_record_list.append(weight_record)
        record = test_validation(valid_data_loader, model, num_classes, aggregation_weight, device)    
        performance_record_list.append(record)

    print('\n')            
    print('='*25, ' Final results ', '='*25)
    print('\n')
    print('Top-1 accuracy on many-shot, medium-shot, few-shot and all classes:')      
        
    for i in range(trials):
        idx = idx_list[i]
        print(test_distribution_set[idx] + '\t')
        print(*performance_record_list[i])
        
    print('\n')
    print('Aggregation weights of three experts:')    
        
    for i in range(trials):
        idx = idx_list[i]
        print(test_distribution_set[idx] + '\t')
        print(*weight_record_list[i])
        
        
def test_training(train_data_loader,model,  aggregation_weight, optimizer,   num_classes, config, args):
    model.eval() 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    losses = AverageMeter('Loss', ':.4e') 
    progress = ProgressMeter(
        len(train_data_loader),
        [losses]) 
      
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6) 
 
    for i, (data, _) in enumerate(tqdm(train_data_loader)):
        data[0] = data[0].to(device)
        data[1] = data[1].to(device) 
        output0 = model(data[0])
        output1 = model(data[1])
        expert1_logits_output0 =  output0['logits'][:,0,:]
        expert2_logits_output0 = output0['logits'][:,1,:]
        expert3_logits_output0 = output0['logits'][:,2,:] 
        expert1_logits_output1 = output1['logits'][:,0,:]
        expert2_logits_output1 = output1['logits'][:,1,:]
        expert3_logits_output1 = output1['logits'][:,2,:]
        aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
        aggregation_output0 = aggregation_softmax[0].cuda() * expert1_logits_output0 + aggregation_softmax[1].cuda() * expert2_logits_output0 + aggregation_softmax[2].cuda() * expert3_logits_output0
        aggregation_output1 = aggregation_softmax[0].cuda() * expert1_logits_output1 + aggregation_softmax[1].cuda() * expert2_logits_output1 + aggregation_softmax[2].cuda() * expert3_logits_output1
        softmax_aggregation_output0 = F.softmax(aggregation_output0, dim=1) 
        softmax_aggregation_output1 = F.softmax(aggregation_output1, dim=1)
        
        # SSL loss: similarity maxmization
        cos_similarity = cos(softmax_aggregation_output0, softmax_aggregation_output1).mean()
        ssl_loss =  cos_similarity 
        loss = - ssl_loss 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(ssl_loss, data[0].shape[0])
         
    aggregation_softmax = torch.nn.functional.softmax(aggregation_weight, dim=0).detach().cpu().numpy()

    return  np.round(aggregation_softmax[0], decimals=2), np.round(aggregation_softmax[1], decimals=2), np.round(aggregation_softmax[2], decimals=2)

def test_validation(data_loader, model, num_classes, aggregation_weight, device):
    model.eval()  
    aggregation_weight.requires_grad = False
    b = np.load("./data/imagenet_lt_shot_list.npy")
    many_shot = b[0]
    medium_shot = b[1] 
    few_shot = b[2]
    confusion_matrix = torch.zeros(num_classes, num_classes).cuda()
    total_logits = torch.empty((0, num_classes)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            expert1_logits_output = output['logits'][:,0,:] 
            expert2_logits_output = output['logits'][:,1,:]
            expert3_logits_output = output['logits'][:,2,:]
            aggregation_softmax = torch.nn.functional.softmax(aggregation_weight) # softmax for normalization
            aggregation_output = aggregation_softmax[0] * expert1_logits_output + aggregation_softmax[1] * expert2_logits_output + aggregation_softmax[2] * expert3_logits_output
            for t, p in zip(target.view(-1), aggregation_output.argmax(dim=1).view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            total_logits = torch.cat((total_logits, aggregation_output))
            total_labels = torch.cat((total_labels, target))  
            
 
    probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)

    # Calculate the overall accuracy and F measurement
    eval_acc_mic_top1= mic_acc_cal(preds[total_labels != -1],
                                        total_labels[total_labels != -1])
         
    acc_per_class = confusion_matrix.diag()/confusion_matrix.sum(1)
    acc = acc_per_class.cpu().numpy() 
    many_shot_acc = acc[many_shot].mean()
    medium_shot_acc = acc[medium_shot].mean()
    few_shot_acc = acc[few_shot].mean()
    print("Many-shot {0:.2f}, Medium-shot {1:.2f}, Few-shot {2:.2f}, All {3:.2f}".format(many_shot_acc * 100, medium_shot_acc * 100,  few_shot_acc * 100, eval_acc_mic_top1* 100))     
    return np.round(many_shot_acc * 100, decimals=2), np.round(medium_shot_acc * 100, decimals=2), np.round(few_shot_acc * 100, decimals=2), np.round(eval_acc_mic_top1 * 100, decimals=2)


def fix_numpy_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

        
if __name__ == '__main__':
    default_ir = 1 / 50
    
    args = argparse.ArgumentParser(description='PyTorch Template')
 
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--epochs', default=1, type=int,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('--ir', default=default_ir, type=float)
    args.add_argument('--seed', default=0, type=int)
    
    config = ConfigParser.from_args(args, test=True)
    main(config, args)