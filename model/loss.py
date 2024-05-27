import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
eps = 1e-7 

def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, cls_num_list=None, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
    
    def _hook_before_epoch(self, epoch):
        pass

    def forward(self, output_logits, target):
        return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)

class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        
        return self

    def forward(self, output_logits, target): # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # CB loss
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)  # * class number
                # the effect of per_cls_weights / np.sum(per_cls_weights) can be described in the learning rate so the math formulation keeps the same.
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot index
         
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1)) 
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s 

        final_output = torch.where(index, x_m, x) 
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)   # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor   #Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)    # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:  
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss
  
 
class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None,  max_m=0.5, s=30, tau=2):
        super().__init__()
        self.base_loss = F.cross_entropy 
     
        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau 

    def inverse_prior(self, prior): 
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0]-1-idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0

        # Obtain logits from each expert  
        expert1_logits = extra_info['logits'][0]
        expert2_logits = extra_info['logits'][1] 
        expert3_logits = extra_info['logits'][2]  
 
        # Softmax loss for expert 1 
        loss += self.base_loss(expert1_logits, target)
        
        # Balanced Softmax loss for expert 2 
        expert2_logits = expert2_logits + torch.log(self.prior + 1e-9) 
        loss += self.base_loss(expert2_logits, target)
        
        # Inverse Softmax loss for expert 3
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9) 
        loss += self.base_loss(expert3_logits, target)
   
        return loss
            
        
class LogitAdjustExpertLoss(nn.Module):
    def __init__(self, cls_num_list=None, lambda_var=1.0, power=1.0, trials=60, loss_type='LA', alpha_sum=10000, dataset_type='CIFAR'):
        super().__init__()
        
        self.cls_num_list = cls_num_list
        
        self.prior = torch.tensor(cls_num_list).float().cuda()
        self.prior = self.prior / self.prior.sum()
        self.C_number = len(cls_num_list)
        self.power = power
        self.lambda_var = lambda_var
        self.trials = trials
        
        self.loss_type = loss_type
        self.few_shot_classes = None
        
        self.alpha_sum = alpha_sum
        self.dataset_type = dataset_type
        
        self.loss = nn.CrossEntropyLoss()
        
    def set_few_shot_classes(self, few_shot_classes):
        self.few_shot_classes = few_shot_classes
        
        if hasattr(self.loss, "set_few_shot_classes"):
            self.loss.set_few_shot_classes(few_shot_classes)
    
    def start_new_epoch(self, epoch, total_epoch):
        if epoch <= int(0.8 * total_epoch):
            self.flag = True
        else:
            self.flag = False
            
        np.random.shuffle(self.pool)
        self.idx = 0
        
    def generate_weights(self):
        if self.dataset_type == 'CIFAR':
            weight = torch.tensor([1 / self.C_number] * self.C_number).float().cuda()
        
            # expert for long-tail distribution
            self.weights1 = self.prior.clone()
            
            # expert for uniform distribution
            self.weights2 = weight
            
            # expert for inverse long-tail distribution
            self.weights3 = self.inverse_prior(self.prior) 
            
        elif self.dataset_type == 'ImageNet':
            weight = torch.tensor([1 / self.C_number] * self.C_number).float().cuda()
            
            # expert for long-tail distribution
            self.weights1 = self.prior.clone()
            
            # expert for uniform distribution
            self.weights2 = weight
            
            # expert for inverse long-tail distribution
            sorted_idx = np.argsort(self.cls_num_list)
            sorted_train_cls_num_list = np.sort(self.cls_num_list)
            sorted_idx = sorted_idx[::-1]
            img_num_per_cls = [0] * self.C_number
            for i in range(self.C_number):
                img_num_per_cls[sorted_idx[i]] = sorted_train_cls_num_list[i]
            
            self.weights3 = torch.tensor(img_num_per_cls).float().cuda()        
            
        else:
            raise NotImplementedError()
        
        # normalize the weights
        self.weights1 = self.weights1 / self.weights1.sum()
        self.weights2 = self.weights2 / self.weights2.sum()
        self.weights3 = self.weights3 / self.weights3.sum()    
        
        
    def generate_weights_pool(self, batch_num):
        size = self.trials * batch_num
        
        self.generate_weights()
        
        self.pool = []
        
        self.idx = 0
        
        weight_list = [self.weights1.cpu().numpy(), self.weights2.cpu().numpy(), self.weights3.cpu().numpy()]
        
        for _ in range(size):
            k = np.random.randint(len(weight_list))
            alpha_temp = weight_list[k] * self.alpha_sum
            dirichlet = np.random.dirichlet(alpha_temp)
            self.pool.append((torch.tensor(dirichlet).float().cuda(), k))   
            
    def get_weights_pool(self):
        return self.pool
                
    def inverse_prior(self, prior):
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0] - 1 - idx1 # reverse the order
        inverse_prior = value.index_select(0,idx2)
        
        return inverse_prior
    
    def dirichlet_mixture(self, weight_list, trials):
        k_list = []
        dirichlet_list = []
        
        for t in range(trials):
            k = np.random.randint(len(weight_list))
            k_list.append(k)
            alpha_temp = weight_list[k] * 10000
            alpha_temp = alpha_temp.cpu().numpy()
            dirichlet = np.random.dirichlet(alpha_temp)
            dirichlet_list.append(dirichlet)
        
        return k_list, dirichlet_list
               
    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return F.cross_entropy(output_logits, target)
        
        t_list = self.pool[self.idx: self.idx + self.trials]
        index = []
        diri = []
        for t in t_list:
            index.append(t[1])
            diri.append(t[0])
            
        self.idx += self.trials
        
        expert_logits = extra_info['logits']
        loss_list = []
    
        for (i, w) in zip(index, diri):
            pi_ratio = w ** self.power / self.prior ** self.power
            expert  = expert_logits[i]
            score = expert - torch.log(pi_ratio + 1e-9)
            
            loss = self.loss(score, target)
            loss_list.append(loss) 
              
        loss_tensor = torch.stack(loss_list)
        
        loss_mean = loss_tensor.mean()
        loss_var = 0.0
        cnt = 0
        for i in range(loss_tensor.size(0)):
            loss_tmp = loss_tensor[i]
            if loss_tmp > loss_mean:
                loss_var += (loss_tmp - loss_mean) ** 2
                cnt += 1
                
        loss_var = loss_var / max(cnt - 1, 1)
        
        if self.flag == True:
            return loss_tensor.mean()
        else:
            return loss_tensor.mean() + self.lambda_var * loss_var
        
