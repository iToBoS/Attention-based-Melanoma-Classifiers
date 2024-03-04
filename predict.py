import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from dataset import get_transforms, MelanomaDataset
import models
from train import get_trans
import wandb
from torchinfo import summary
from test_utils import read_clinical,read_prove

os.environ['WANDB_API_KEY'] = ''
wandb_logger = wandb.init(entity="", project="",
                          dir="", resume=False)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data/ProveAI-TTA') # Prove-processed ProveAI-TTA
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=20)
    parser.add_argument('--out-dim', type=int, default=8)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./home/sana/SIIM-ISIC/test_weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--sub-dir', type=str, default='./subs')
    parser.add_argument('--eval', type=str, choices=['best', 'final','epoch10','epoch20','epoch30'], default="best")
    parser.add_argument('--n-test', type=int, default=10)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')
    parser.add_argument('--hard_samples', action='store_true')
    parser.add_argument('--fold', type=str, default='0')
    args, _ = parser.parse_known_args()
    return args

def main():
    print(args)
    df_test = pd.read_csv(os.path.join(args.data_dir, 'test.csv')) #'test.csv'
    df_test['image_name'] = df_test['isic_id']
    print("len df test", len(df_test))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(args.data_dir, f'{x}.jpg'))
    print(df_test['filepath'][0])

    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('SEK', 'BKL'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('ACK', 'AK'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('actinic keratosis', 'AK'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('NEV', 'NV'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('nevus', 'NV'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('dermatofibroma', 'DF'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('squamous cell carcinoma', 'SCC'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('basal cell carcinoma', 'BCC'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('dermatofibroma', 'DF'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('dermatofibroma', 'DF'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'NV'))
    #unknown classes:
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('verruca', 'unknown'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('angioma', 'VASC'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('angiofibroma or fibrous papule', 'unknown'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('clear cell acanthoma', 'unknown'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('scar', 'unknown'))
   
    print("diagnosis",set(df_test['diagnosis']))

    #df_test= df_test.loc[df_test["diagnosis"] =='unknown']
    diagnosis2idx = {'AK':0,'BCC':1, 'BKL':2 , 'DF':3 , 'SCC':4 , 'VASC':5, 'melanoma':6 , 'NV':7 ,'unknown': 8}

    df_test['target'] = df_test['diagnosis'].map(diagnosis2idx)
    print(df_test['target'])
    meta_features = None
    n_meta_features = 0
    mel_idx = 6

    if args.hard_samples:
        df_test = df.reset_index(drop=True)

    transforms_test= get_transforms(args.image_size,get_test=True)

    if args.DEBUG:
        df_test = df_test.sample(100) 
    print("Prove set: \n",df_test['diagnosis'].value_counts())
    dataset_test = MelanomaDataset(df_test, 'test', meta_features=meta_features, transform=transforms_test) 
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # load model
    models = []
    print("len test set", len(dataset_test))
    folds = [int(i) for i in args.fold.split(',')]
    for fold in folds:
        print(args.eval)
        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
        if args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')
        if args.eval == 'epoch10':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_epoch10_fold{fold}.pth')
        if args.eval == 'epoch20':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_epoch20_fold{fold}.pth')
        if args.eval == 'epoch30':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_epoch30_fold{fold}.pth')
               
        new = False
        print(args.enet_type)
       
        if args.enet_type != 'tf_efficientnet_b3':
            model = ModelClass( out_dim = args.out_dim,
                            pretrained=False)
        else:
            model = ModelClass(
                args.enet_type,
                n_meta_features=n_meta_features,
                n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
                out_dim=args.out_dim,
                pretrained=False)

        model = model.to(device)
        summary(model, input_size=(1, 3, args.image_size, args.image_size))
      
        if new:
            try:  # single GPU model_file
                model.load_state_dict(torch.load(model_file)['model_state_dict'], strict=False)
            except:  # multi GPU model_file
                state_dict = torch.load(model_file)
                state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
                model.load_state_dict(state_dict['model_state_dict'], strict=False)
        #if args.eval== 'external':


        else:
            try:  # single GPU model_file
                model.load_state_dict(torch.load(model_file), strict=True)
            except:  # multi GPU model_file
                state_dict = torch.load(model_file)
                state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
                model.load_state_dict(state_dict, strict=True)  

        if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()
        models.append(model)

    # predict
    PROBS = []

    with torch.no_grad():
        for (data) in tqdm(test_loader):
                if args.use_meta:
                    data, meta = data
                    data, meta = data.to(device), meta.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for model in models:
                        for I in range(args.n_test):
                            l = model(get_trans(data, I), meta)
                            probs += l.softmax(1)
                else:   
                    data = data.to(device)
                    probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                    for model in models:
                        for I in range(args.n_test):
                            l = model(get_trans(data, I))
                            probs += l.softmax(1)

                probs /= args.n_test
                probs /= len(models)
                PROBS.append(probs.detach().cpu())
    PROBS = torch.cat(PROBS).numpy()
    
    PROBS=pd.DataFrame(PROBS)
    PROBS['image_name'] = df_test['image_name'].reset_index(drop=True)
    PROBS.to_csv(os.path.join(args.sub_dir, f'probs_{args.enet_type}_{args.eval}_prove.csv'), index=False)

    TTA = False
    if 'TTA' in args.data_dir:
        TTA= True
    print("TTA:",TTA)
    if TTA:
        PROBS=read_prove(PROBS,df_test['target'].reset_index(drop=True), classes_n= args.out_dim)
    else:
        PROBS=read_clinical(PROBS,y_test=df_test['target'],classes_n= args.out_dim) 
    print(PROBS)
    print(f"shape of probs: {probs.shape}")
    subs =np.column_stack(( PROBS[:, args.out_dim],PROBS[:, mel_idx]))
    df = pd.DataFrame(subs, columns=['image', 'malignant'])
    print(f"len probs: {len(PROBS)} and len df test {len(df_test)}")
    df.to_csv(os.path.join(args.sub_dir, f'sub_{args.enet_type}_{args.eval}_prove.csv'), index=False)
    PROBS=pd.DataFrame(PROBS)
    PROBS.to_csv(os.path.join(args.sub_dir, f'probs_{args.enet_type}_{args.eval}_prove.csv'), index=False)
if __name__ == '__main__':

    args = parse_args()
    wandb.config.update(args)
    os.makedirs(args.sub_dir, exist_ok=True)
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

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1
    device = torch.device('cuda')

    main()
