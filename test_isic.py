import os
import torch
import numpy as np
from scipy.stats import rankdata as rd
from dataset_multi_output import get_df, get_transforms, MelanomaDataset
from ISIC2020_winners.second.src.factory.models import classifier  # Adjust the import based on your module structure
from tqdm import tqdm
import tensorflow as tf
import efficientnet.tfkeras as efn
import pandas as pd
weights = '11' # '3' '11' '2'
UNITS = False
os.environ['CUDA_VISIBLE_DEVICES'] = ' MIG-cdc1351f-1b7a-554c-a273-f7643f99523f'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_prove():
    print("prove test set :")
    data_dir = '/home/falcon/sana/scratch/Classifier/data/ProveAI-TTA'
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv')) #'test.csv'
    df_test['image_name'] = df_test['isic_id']
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'{x}.jpg'))
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
    #unknown classes:
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'NV'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('verruca', 'unknown'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('angioma', 'VASC'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('angiofibroma or fibrous papule', 'unknown'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('clear cell acanthoma', 'unknown'))
    df_test['diagnosis']  = df_test['diagnosis'].apply(lambda x: x.replace('scar', 'unknown'))
    #df_test= df_test.loc[df_test["diagnosis"] =='unknown']
    print("counts of classes:", df_test['diagnosis'].value_counts())
    diagnosis2idx = {'AK':0,'BCC':1, 'BKL':2 , 'DF':3 , 'SCC':4 , 'VASC':5, 'melanoma':6 , 'NV':7 , 'unknown': 8}
    df_test['target'] = df_test['diagnosis'].map(diagnosis2idx)
    #df_test.to_csv('prove_unknown.csv', index=False)
    malben_idx = {'malignant':1 , 'benign':0, 'indeterminate/benign': 0 ,'indeterminate/malignant': 1 } 
    df_test['benign_malignant'] = df_test['benign_malignant'].map(malben_idx)
    return df_test
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
def get_2nd_models():
    weights_dir =["./ISIC2020_winners/second/checkpoints/bee508",
                   "./ISIC2020_winners/second/checkpoints/bee517",
                   "./ISIC2020_winners/second/checkpoints/bee608"]
    models0 = []
    models1 = []
    models2 = []
    # Initialize the model
    model0 = classifier.Net2D(backbone='tf_efficientnet_b6_ns', pretrained=False, num_classes=3, dropout=0.2, multisample_dropout=True)
    model1 = classifier.Net2D(backbone='tf_efficientnet_b7', pretrained=False, num_classes=3, dropout=0.2, multisample_dropout=True)
    model2 = classifier.Net2D(backbone='tf_efficientnet_b6_ns', pretrained=False, num_classes=3, dropout=0.2, multisample_dropout=True)
    print("all models imported!")
    model0.to(device)
    model1.to(device)
    model2.to(device)
    # Loop through each fold and load the corresponding weights
    print("laoding weights... ")
    for fold in range(5):
        weights_path0 = os.path.join(weights_dir[0], f"fold{fold}.PTH")
        weights0 = torch.load(weights_path0, map_location=lambda storage, loc: storage)
        weights0 = {k.replace('module.', '') : v for k,v in weights0.items()}
        model0.load_state_dict(weights0)
        model0.eval()
        models0.append(model0)
 
        weights_path1 = os.path.join(weights_dir[1], f"fold{fold}.PTH")
        weights1 = torch.load(weights_path1, map_location=lambda storage, loc: storage)
        weights1 = {k.replace('module.', '') : v for k,v in weights1.items()}
        model1.load_state_dict(weights1)
        model1.eval()
        models1.append(model1)
       
        weights_path2 = os.path.join(weights_dir[2], f"fold{fold}.PTH")
        weights2 = torch.load(weights_path2, map_location=lambda storage, loc: storage)
        weights2 = {k.replace('module.', '') : v for k,v in weights2.items()}
        model2.load_state_dict(weights2)
        model2.eval()
        models2.append(model2)
 
    return models0, models1, models2

def predict_torch(models,test_loader,n_test=10):
        PROBS = []
        with torch.no_grad():
            for (data,targets) in tqdm(test_loader):
                    data = data.to(device)
                    probs = torch.zeros((data.shape[0], 3)).to(device)
                    for model in models:
                        for I in range(n_test):
                            l = model(get_trans(data, I))
                            probs += l.softmax(1)
                    probs /= n_test
                    probs /= len(models)
                    PROBS.append(probs.detach().cpu())
        PROBS = torch.cat(PROBS).numpy()
        PROBS=pd.DataFrame(PROBS)
        return PROBS
def get_3rd_models(net_size,model_type): 
    model_input = tf.keras.Input(shape=(net_size, net_size, 3), name='imgIn')
    dummy = tf.keras.layers.Lambda(lambda x:x)(model_input)
    outputs = []
    constructor = getattr(efn, model_type)
    x = constructor(include_top=False, weights='imagenet', input_shape=(net_size, net_size, 3), pooling='avg')(dummy)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    outputs.append(x)
    model = tf.keras.Model(model_input, outputs, name='aNetwork')
    model.summary() 
    return model
if UNITS:
    print("Predictions using UNITS test set:")
    df, df_test, meta_features, n_meta_features, mel_idx,df_synth = get_df(
            'tf_efficientnet_b6',8,'./data',
            512,False,ISIC2020_test=False)
    output_path = '/scratch/sana/Classifier/ISIC2020_winners/predictions/UNITS/'
else: #Prove dataset
    df_test = get_prove()
    output_path = '/scratch/sana/Classifier/ISIC2020_winners/predictions/prove/'


if weights == '2':
   #label map:nevi:1 , melanoma: 2
   _ ,transforms_val = get_transforms(384)
   dataset_test = MelanomaDataset(df_test, 'test', meta_features=None, transform=transforms_val) 
   test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=128, num_workers=2)
   print("Predictions using 2nd winner model:")
   models0, models1, models2= get_2nd_models()

   print("First model:")
   probs0 = predict_torch(models0, test_loader)
   probs0['image_name'] = df_test['image_name']
   probs0.to_csv(output_path+'2nd-b6-1.csv', index=False)

   print("Second model:")
   probs1 = predict_torch(models1, test_loader)
   probs1['image_name'] = df_test['image_name']
   probs1.to_csv(output_path+'2nd-b7.csv', index=False)

   print("Third model:")
   probs2 = predict_torch(models2, test_loader)
   probs2['image_name'] = df_test['image_name']
   probs2.to_csv(output_path+'2nd-b6-2.csv', index=False)

elif weights == '3':
    print("Predictions using 3rd winner model:")
    base_path = './ISIC2020_winners/third/'
    models = []
    files = ['EfficientNetB6_256x256_2019-2020_epoch25_auc_0.95.h5',
              'EfficientNetB6_256x256_2020_epoch13_auc_0.92.h5',
              'EfficientNetB6_384x384_2019-2020_epoch25_auc_0.97.h5',
              'EfficientNetB6_384x384_2020_epoch15_auc_0.96.h5',
              'EfficientNetB6_512x512_2019-2020_epoch12_auc_0.97.h5',
              'EfficientNetB6_512x512_2020_epoch15_auc_0.96.h5',
              'EfficientNetB6_768x768_2019-2020_epoch10_auc_0.97.h5',
              'EfficientNetB6_768x768_2020_epoch15_auc_0.96.h5']
    sizes = [256,256,384,384,512,512,768,768]
    for i in range(len(files)):
        model_path = os.path.join(base_path,files[i])
        model = get_3rd_models(sizes[i],model_type = 'EfficientNetB6')
        model.load_weights(model_path)
        models.append(model)
    print("\n len models:",len(models))
    for i in range(len(models)):
        print(f"model {i}:")
        _ ,transforms_val = get_transforms(sizes[i])
        dataset_test = MelanomaDataset(df_test, 'test', meta_features=None, transform=transforms_val) 
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=128, num_workers=2)
        all_preds = []  # List to accumulate predictions
        for (data, targets) in tqdm(test_loader):
            data = data.numpy()
            data = np.transpose(data, (0, 2, 3, 1))
            preds = np.zeros((data.shape[0], 1))
            for I in range(10):
                preds += models[i].predict(data, verbose=1, steps=10)
            preds /= 10
            all_preds.append(preds)
        # Concatenate predictions from all batches
        all_preds = np.concatenate(all_preds, axis=0)
        preds_df = pd.DataFrame(all_preds) 
        preds_df.to_csv(output_path + f'10n-3rd-b6-{i}.csv', index=False) 
        print(f"model {i} done")
elif weights == '11':
    df_test.to_csv(output_path + f'prove.csv', index=False) 
    pass
 