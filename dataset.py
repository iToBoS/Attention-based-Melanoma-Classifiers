import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]
     
    def get_labels(self):
        return torch.tensor(self.csv['target']).float()
    def __getitem__(self, index):

        #print("mod:",self.mode)
        row = self.csv.iloc[index]
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data          
        else:
            label = torch.tensor(self.csv.iloc[index].target).long()
            return data, label
     
def get_transforms(image_size,get_test = False):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])
    
    transforms_test = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])
    
    transforms_val = transforms_train
    
    if get_test:
        return transforms_test

    return transforms_train, transforms_val

def get_meta_data(df_train, df_test):
    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
    # Sex features
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)
    # Age features
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)
    # n_image per user
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)
    # image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
        train_sizes[i] = os.path.getsize(img_path)
    df_train['image_size'] = np.log(train_sizes)
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)

    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features


def get_df(kernel_type, out_dim, data_dir, data_folder,dca=True , use_meta=False,complete_synth=False,add_fake=False,ISIC2020_test=False):
    #sana
    Seven_checkpoint= False
    sample_unk = False
    homography = False
    synthetic_data = False #to combine some fake samples into real onws
    if complete_synth:
        synthetic_data= True #for this to be fully active you need to set synthetic_data = True
    df_synth = []
    # 2020 data
    df_train = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'train.csv'))
    df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/train', f'{x}.jpg'))
    df_train['is_ext'] = 0
    
    if Seven_checkpoint:
        df_checkp = pd.read_csv(os.path.join(data_dir,'7-checkpoint/7checkpoint_meta.csv')).reset_index(drop=True) 
        df_checkp['is_ext'] = 4
        df_checkp['tfrecord']= 1 
        df_checkp['filepath'] = df_checkp['derm'].apply(lambda x: os.path.join(data_dir, '7-checkpoint/derm', f'{x}'))
        df_train = pd.concat([df_train, df_checkp]).reset_index(drop=True) 
        

    # if homography:
    #     print("homography activated")  
    #     hom_path = 'homography_data'
    #     df_hom =pd.read_csv(os.path.join(data_dir, 'homography_data/train.csv'))
    #     df_hom['filepath'] = df_hom['image_name'].apply(lambda x: os.path.join(data_dir, 'homography_data/train', f'{x}.jpg'))
    #     df_hom['diagnosis'] = df_hom['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    #     df_hom['diagnosis'] = df_hom['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    #     df_hom['fold'] = df_hom['tfrecord'] 
    #     df_hom['is_ext'] = 1
    
    # if synthetic_data:
    #     sample = True
    #     print("sythetic data added") 
    #     synth_path = 'synthetic'        
    #     df_synth = pd.read_csv(os.path.join(data_dir, synth_path+'/train.csv')) 
    #     df_synth = df_synth.loc[df_synth["diagnosis"] !='NV']
    #     df_synth['filepath']  = df_synth['image_name'].apply(lambda x: os.path.join(data_dir, synth_path+'/train', f'{x}.jpg'))
    #     df_synth['diagnosis'] = df_synth['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    #     df_synth['diagnosis'] = df_synth['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))       
    #     df_synth['fold'] = df_synth['tfrecord'] 
    #     df_synth['is_ext'] = 3
        #sample each class
    #     if sample:
    #         df_sample = pd.DataFrame()
    #         groups =df_synth.groupby('diagnosis')
    #         for name, group in groups:
    #             print(name)
    #             if name in ['DF','VASC','SCC']:
    #                 df_sample = df_sample.append(group.sample(n=5700, random_state=42)) 
    #             elif name=='AK':
    #                 df_sample = df_sample.append(group.sample(n=5200, random_state=42))  
    #             elif name =='BKL' :
    #                 df_sample = df_sample.append(group.sample(n=3200, random_state=42)) 
    #             elif name =='BCC' :
    #                 df_sample = df_sample.append(group.sample(n=2700, random_state=42))                 
    #             else:
    #                 df_sample = df_sample.append(group.sample(n=1000, random_state=42))  
            
    #         df_synth = df_sample
    #         print(f"len df_synth: {len(df_synth)}")
        
    # if add_fake:
    #     add_all = False
    #     print("sythetic data added") 
    #     synth_path = 'synthetic'     
    #     synth_path2 = 'synthetic-old'   
    #     df_synth = pd.read_csv(os.path.join(data_dir, synth_path+'/train.csv')) 
    #     df_synth2 = pd.read_csv(os.path.join(data_dir, synth_path2+'/train.csv')) 
    #     df_synth = df_synth.loc[df_synth["diagnosis"] !='NV']
    #     #df_synth = df_synth.loc[df_synth["diagnosis"] !='AK'] #we drop the AK samples from synth and use the AK samples from old synth
    #     df_synth['filepath']  = df_synth['image_name'].apply(lambda x: os.path.join(data_dir, synth_path+'/train', f'{x}.jpg'))

    #     df_synth2['filepath']  = df_synth2['image_name'].apply(lambda x: os.path.join(data_dir, synth_path2+'/train', f'{x}.jpg'))
    #     df_synth['diagnosis'] = df_synth['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))  
    #     if add_all:    
    #         df_synth = pd.concat([df_synth,df_synth2]).reset_index(drop=True) 
    #     df_synth['fold'] = df_synth['tfrecord'] 
    #     df_synth['is_ext'] = 3

    df_train['fold'] = df_train['tfrecord'] ##added by sana   

    # 2018, 2019 data (external data)
    df_train2 = pd.read_csv(os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}', 'train.csv'))
    df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
    df_train2['filepath'] = df_train2['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}/train', f'{x}.jpg'))

    df_train2['fold'] = df_train2['tfrecord'] ##added by sana
    df_train2['is_ext'] = 1 ##added by sana - is it external data?
    # Preprocess Target
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))   
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))
    #if out_dim == 9: #sana changed it from 9 to 8
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
    df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    if out_dim == 4:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('DF', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('AK', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('SCC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('VASC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('BCC', 'unknown'))
 
    if out_dim==8:
        df_train = df_train.loc[df_train["diagnosis"] !='unknown']
    elif sample_unk:
            rows_to_drop = df_train.loc[df_train["diagnosis"] =='unknown'].sample(5000).index
            df_train = df_train.drop(rows_to_drop).reset_index(drop=True) 
    
    print("isic 2020: \n",df_train['diagnosis'].value_counts())
    print("isic 2020- total: \n",len(df_train))
    print("isic 2019: \n ",df_train2['diagnosis'].value_counts())
    print("isic 2019- total: \n",len(df_train2))
    # concat train data    
    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True) 

    if dca:
        print("\n dca is active! \n")
        dca_2020_dir = '/home/falcon/sana/scratch/dark_corner_artifact_removal/Modules/processed_data/2020'
        dca_2019_dir = '/home/falcon/sana/scratch/dark_corner_artifact_removal/Modules/processed_data/2019'
        file_names_2020 = os.listdir(dca_2020_dir)
        file_names_2019 = os.listdir(dca_2019_dir)

        print("len dca files:",len(file_names_2019), len(file_names_2020))

        for index, row in df_train.iterrows():
            file_name = row['image_name'] +'.jpg'#.replace('.jpg', '')
            
            # Check if the file name exists in the directory
            if file_name in file_names_2020:
                file_path = os.path.join(dca_2020_dir, file_name) 
                df_train.at[index, 'filepath'] = file_path

            elif file_name in file_names_2019:
                file_path = os.path.join(dca_2019_dir, file_name) 
                df_train.at[index, 'filepath'] = file_path

    # if homography:
    #     df_train = pd.concat([df_train, df_hom]).reset_index(drop=True)

        #update kfold                       
    # test data
    if ISIC2020_test:
        df_test = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'isic2020_test_augmented.csv'))
        df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/isic2020_test_augmented', f'{x}.jpg'))
    else:
        df_test = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'test.csv'))
        df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/test', f'{x}.jpg'))

    # if synthetic_data:
    #     if not complete_synth:
    #         df_train = pd.concat([df_train, df_synth]).reset_index(drop=True)
    #     else:
    #         df_train = df_synth
    #         df_train = generate_kfold(df_train,data_dir=data_dir,Path=synth_path)
    #update kfold    
        
    if use_meta:
        df_train, df_test, meta_features, n_meta_features = get_meta_data(df_train, df_test)
    else:
        meta_features = None
        n_meta_features = 0

    # class mapping
    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
    print(diagnosis2idx)
    df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
    mel_idx = diagnosis2idx['melanoma']
    print(f"mel_idx is : {mel_idx}")
 
    
    return df_train, df_test, meta_features, n_meta_features, mel_idx, df_synth
