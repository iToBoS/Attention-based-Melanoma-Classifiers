import os
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score,f1_score ,balanced_accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from util import GradualWarmupSchedulerV2,EarlyStopper
import apex
from apex import amp
from dataset import get_df, get_transforms, MelanomaDataset
import models
from loss import FocalLoss,LMFLoss,LDAMLoss
import wandb
from torchinfo import summary
from val_utils import read_classification


os.environ['WANDB_API_KEY'] = '706ff0f749dd4eb7c95e920de4c29f5a0de06d94'
wandb_logger = wandb.init(entity="sana-nz", project="classifier-2",
                          dir="/home/falcon/sana/scratch/Classifier", resume=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str,required=True) 
    parser.add_argument('--load_model_dir', type=str, default='./home/falcon/sana/SIIM-ISIC/test_weights/swin_fake_mlp2')
    parser.add_argument('--load-kernel-type', type=str, default='8c_swinv2_224_224_18ep') #required=True
    parser.add_argument('--data-dir', type=str, default='./data/')
    parser.add_argument('--data-folder', type=int, default=512)
    parser.add_argument('--image-size', type=int, default=384)
    parser.add_argument('--enet-type', type=str,  default='tf_efficientnet_b3') 
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--init-lr', type=float, default=3e-5) #
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--n-epochs', type=int,default=35) #  default=18
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./home/falcon/sana/SIIM-ISIC/test_weights/')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='MIG-cdc1351f-1b7a-554c-a273-f7643f99523f')
    parser.add_argument('--fold', type=str, default='0,1,2,3,4') 
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--Continue', action='store_true')
    parser.add_argument('--freeze_top', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--fake_train', action='store_true') #when you want to train a model with completely fake samples
    parser.add_argument('--add_fake', action='store_true') #when you want to add fake samples to the real datset
    args, _ = parser.parse_known_args()
    return args

def class_weights(class_counts):
    total_count = np.sum(class_counts)
    class_props = class_counts / total_count
    class_weights = 1.0 / class_props
    class_weights /= np.sum(class_weights)
    return class_weights

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_epoch(model, loader, optimizer):
    model.train()
    train_loss = []
    correct = 0    
    total = 0 
    TARGETS = []
    PROBS = []
    bar = tqdm(loader)
    for (data, target) in bar:

        optimizer.zero_grad()
        
        if args.use_meta:
            data, meta = data
            data, meta, target = data.to(device), meta.to(device), target.to(device)
            logits = model(data, meta)
        else:
            data, target = data.to(device), target.to(device)
            logits = model(data)        
  
        loss = criterion(logits, target)

        if not args.use_amp:
            loss.backward()
        else:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        if args.image_size in [896,576]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)

        for idx, i in enumerate(logits): 
            PROBS.append(torch.argmax(i).cpu())
            TARGETS.append(target[idx].cpu())
            if torch.argmax(i) == target[idx]:
                correct += 1
            total += 1      
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    train_loss = np.mean(train_loss)
    train_f1 = f1_score(TARGETS, PROBS, average='weighted')
    accuracy = round(correct/total, 3)*100
    return train_loss ,accuracy, train_f1 

def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)


def val_epoch(model, loader, mel_idx, is_ext=None, n_test=10, get_output=False):
    model.eval()
    auc_20 = 0
    bacc_19 = 0
    val_loss = []
    LOGITS = []
    PROBS = []
    TARGETS = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):           
            if args.use_meta:
                data, meta = data
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I), meta)
                    logits += l
                    probs += l.softmax(1)
            else:
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    logits += l
                    probs += l.softmax(1)
            logits /= n_test
            probs /= n_test

            LOGITS.append(logits.detach().cpu())
            PROBS.append(probs.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits, target)
            val_loss.append(loss.detach().cpu().numpy())

    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    

    if get_output:
        return LOGITS, PROBS
    else:
        acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
        val_f1 = f1_score(TARGETS, PROBS.argmax(1), average='weighted')
        #mel-auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
        auc = roc_auc_score(TARGETS,PROBS, multi_class='ovr')
        if not args.fake_train:
            auc_20 = roc_auc_score((TARGETS[is_ext == 0] == mel_idx).astype(float), PROBS[is_ext == 0, mel_idx])
            #print("target:",TARGETS[is_ext == 1].shape)
            #print("probs:",PROBS[is_ext == 1][:, :-1].shape)
            #auc_19 = roc_auc_score((TARGETS[is_ext == 1]), PROBS[is_ext == 1][:, :-1],multi_class='ovr')
            bacc_19 = balanced_accuracy_score((TARGETS[is_ext == 1]), PROBS[is_ext == 1].argmax(1))

        PROBS=pd.DataFrame(PROBS)
        read_classification(PROBS,TARGETS,args.out_dim)
        return val_loss, acc, auc, val_f1 ,auc_20,bacc_19#,auc_19


def run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx,df_synth,n):

    if args.DEBUG:
        args.n_epochs = 3
        df_train = df[df['fold'] != fold].sample(args.batch_size * 5)
        df_valid = df[df['fold'] == fold].sample(args.batch_size * 2)
    else:
        df_train = df[df['fold'] != fold]
        df_valid = df[df['fold'] == fold]

    dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
    dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers) #was r=RandomSampler
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)
    if not args.fake_train:
        print(f"len df valid for 2020: {df_valid['is_ext'].value_counts()[0]}")
        print(f"len df valid for 2019: {df_valid['is_ext'].value_counts()[1]}")

    if args.enet_type != 'tf_efficientnet_b3':
        model = ModelClass( out_dim = args.out_dim,
                        pretrained=args.pretrained)
    else:
        model = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim,
            pretrained=args.pretrained)

    if DP:
        model = apex.parallel.convert_syncbn_model(model)
    model = model.to(device)
    auc_max = 0.
    #auc_20_max = 0.
    model_file  = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
    model_file_train1  = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}_train.pth')
    #model_file2 = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
    model_file3 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')
    model_file_train2 = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}_train.pth')
    ###############################################################################################################
    #optimizer = optim.Adam(model.parameters(), lr=args.init_lr,weight_decay=args.weight_decay)
    #optimizer = AdamP(model.parameters(), lr=args.init_lr,weight_decay=args.weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=args.init_lr,weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs - 1) 
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler)
    #ep = 0
    if args.finetune or args.Continue: 
        loaded_model = os.path.join(args.load_model_dir, f'{args.load_kernel_type}_best_fold{fold}.pth')
        loaded_model2 = os.path.join(args.load_model_dir, f'{args.load_kernel_type}_final_fold{fold}.pth')
        try:  # single GPU model_file
            model.load_state_dict(torch.load(loaded_model), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(loaded_model)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)

        print(f"pre-trained model loaded ")       
        print("optimizer lr is:", {optimizer.param_groups[0]["lr"]})
     
    if args.finetune and 'swin' not in args.kernel_type:
        in_ch = model.myfc.in_features
        model.myfc =  nn.Linear(in_ch, args.out_dim)
    ###############################################################################################################

    if args.use_amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    if DP:
        model = nn.DataParallel(model)
#     scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs - 1)

    """
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    print("len(dataset_valid):", len(dataset_valid))
    print(model)
    wandb.watch(model)
    summary(model, input_size=(1, 3, args.image_size, args.image_size))
    early_stopper = EarlyStopper()
    for epoch in range(1, args.n_epochs + 1):
        #sample Nv class here
        nv_sample=False   
        if nv_sample:
            nv_samples = df_train[df_train['diagnosis'] == 'nevus'].sample(n=6000)
            df_without_nv = df_train[df_train['diagnosis'] != 'nevus']
            df_train = pd.concat([nv_samples, df_without_nv]).reset_index(drop=True)

        print(f"len  df train: {df_train['diagnosis'].value_counts()}")
        if args.add_fake:
            #remove previous fake samples
            df_train = df_train[df_train['is_ext'] != 3]
            #df_synth = df_synth[df_synth['diagnosis'] != 'BCC']
            #get len each class and sample as the same size
            ak_samples = df_synth[df_synth['diagnosis'] == 'AK'].sample(n=n)
            bcc_samples = df_synth[df_synth['diagnosis'] == 'BCC'].sample(n=n) #bcc_samples add this to  concat
            bkl_samples = df_synth[df_synth['diagnosis'] == 'BKL'].sample(n=n)
            df_samples = df_synth[df_synth['diagnosis'] == 'DF'].sample(n=n)
            scc_samples = df_synth[df_synth['diagnosis'] == 'SCC'].sample(n=n)
            vasc_samples = df_synth[df_synth['diagnosis'] == 'VASC'].sample(n=n)
            mel_samples = df_synth[df_synth['diagnosis'] == 'melanoma'].sample(n=n)
            #df_train = pd.concat([df_train,ak_samples,bkl_samples,df_samples,scc_samples,vasc_samples,mel_samples]).reset_index(drop=True)
            df_train = pd.concat([df_train,df_samples,scc_samples,vasc_samples]).reset_index(drop=True)

        print(f"len final df train: {df_train['diagnosis'].value_counts()}")
        #dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
        #train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=args.num_workers) #was r=RandomSampler
        print(time.ctime(), f'Fold {fold}, Epoch {epoch}')
#         scheduler_warmup.step(epoch - 1)
        #do not forget the accuracy
        train_loss, train_acc , train_f1 = train_epoch(model, train_loader, optimizer)
        val_loss, acc, auc , val_f1,auc_20,bacc_19= val_epoch(model, valid_loader, mel_idx, is_ext=df_valid['is_ext'].values)
        #add train acc and auc to content
        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train_loss: {train_loss:.5f}, train_acc: {train_acc:.5f} ,train_f1: {train_f1:.5f} ,val_f1: {(val_f1):.4f}, valid_loss: {(val_loss):.5f}, val_acc: {(acc):.4f},  Vauc: {(auc):.6f}'
        wandb.log({'Fold': fold,'Epoch': epoch, 'Lr':optimizer.param_groups[0]["lr"], 'train_acc': train_acc, 'train_f1':train_f1, 'train_loss': train_loss, 'valid_loss' : val_loss, 'val_acc' : acc, 'val_f1':val_f1 ,'val_AUC' : auc,'auc20' : auc_20,'bacc_19' : bacc_19})
        print(content)
        with open(os.path.join(args.log_dir, f'log_{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')
       
      
        scheduler_warmup.step()    
        if epoch==2: scheduler_warmup.step() # bug workaround   )
            
        if auc > auc_max:
            print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
            torch.save(model.state_dict(), model_file)
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scheduler_warmup_state_dict':scheduler_warmup.state_dict(),
            'loss': train_loss
            }, model_file_train1)
            auc_max = auc
        """ 
        if epoch%10==0:
            file  = os.path.join(args.model_dir, f'{args.kernel_type}_epoch{epoch}_fold{fold}.pth')
            torch.save(model.state_dict(), file) 
   
        if early_stopper.early_stop(val_loss,val_f1):             
            break
        """
    torch.save(model.state_dict(), model_file3)
    torch.save({
            'epoch': args.n_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scheduler_warmup_state_dict':scheduler_warmup.state_dict(),
            'loss': train_loss
            }, model_file_train2)
    

def main():

    df, df_test, meta_features, n_meta_features, mel_idx,df_synth= get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir,
        args.data_folder,
        complete_synth=args.fake_train,
        add_fake= args.add_fake
    )

    transforms_train, transforms_val = get_transforms(args.image_size)
   
    folds = [int(i) for i in args.fold.split(',')]
    for fold in folds:
        run(fold, df, meta_features, n_meta_features, transforms_train, transforms_val, mel_idx,df_synth,n)
        
   

if __name__ == '__main__':

    args = parse_args()
    print(args)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES
  
    if 'efficientnet' in args.enet_type and 'modified' not in args.enet_type:
        ModelClass = models.Effnet_Melanoma
    elif args.enet_type =='S2MLP_B3': #SA16
        ModelClass = models.S2MLP_B3
    elif args.enet_type =='Spatial_B3': #SA1
        ModelClass = models.Spatial_B3
    elif args.enet_type =='SGE_B3': #SA6
        ModelClass = models.SGE_B3
    elif args.enet_type =='S2MLP_CBAM': #SA27
        ModelClass = models.S2MLP_CBAM
    else:
        raise NotImplementedError()

    print("model class", ModelClass)
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1
    wandb.config.update(args)

    set_seed()
  
    device = torch.device('cuda')

    #({'AK': 0, 'BCC': 1, 'BKL': 2, 'DF': 3, 'SCC': 4, 'VASC': 5, 'MEL': 6, 'NV': 7})
    if args.add_fake:
        n = 2500
    else:
        n=0
    if args.out_dim == 8:
        class_counts = [867,3320,2837,239+n,628+n,253+n,5090,14000]
    elif args.out_dim == 9:
        class_counts = [867,3320,2837,239+n,628+n,253+n,5090,18000,30000]
    print(class_counts)
    c_weights= class_weights(class_counts)
    c_weights = torch.FloatTensor(c_weights).to(device) 
    class_counts = torch.FloatTensor(class_counts).to(device) 

    assert len(class_counts==args.out_dim), 'len class count must be equal to out dim'
    loss_f_type ='LMF' # WCE, LMF, LDAM, FG2
    wandb.log({"Loss Function" : loss_f_type }) #focal , g=2
    #loss function   
    if loss_f_type =='CE':
        criterion = nn.CrossEntropyLoss() 
        
    if loss_f_type =='WCE':
        criterion = nn.CrossEntropyLoss(weight=c_weights) 
        
    if loss_f_type == 'FG2':
        criterion = FocalLoss(alpha=c_weights) 
        
    if loss_f_type == 'LMF':
        criterion = LMFLoss(cls_num_list=class_counts,device=device,alpha=0.2,beta=0.2,weight=c_weights,s=10)

    if loss_f_type == 'LDAM':
        criterion = LDAMLoss(cls_num_list=class_counts,device=device,weight=c_weights,max_m=0.5, s=30)    
                 
    main()
