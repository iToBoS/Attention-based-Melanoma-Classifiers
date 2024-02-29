import pandas as pd
import numpy as np
from glob import glob
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-dir', type=str, default='/scratch/sana/Classifier/ISIC2020_winners/dump') #predictions/prove/third
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    rank_mean = True
    args = parse_args()
    print(args.sub_dir)
    subs = [pd.read_csv(csv) for csv in sorted(glob(os.path.join(args.sub_dir, '*csv')))]
    for i in range(len(subs)):
        df = subs[i]  # Get a reference to the original DataFrame
        if 'first' in args.sub_dir:
            if len(df.columns) < 6:
                subs[i] = df[['image_name', '1','y_true']]  # Modify the DataFrame in place
                subs[i].columns = ['target' if col == '1' else col for col in subs[i].columns]
            
            else:
                subs[i] = df[['image_name', '6','y_true']]  # Modify the DataFrame in place
                subs[i].columns = ['target' if col == '6' else col for col in subs[i].columns]

        elif 'second' in args.sub_dir:     
                subs[i] = df[['image_name', '2','y_true']]   
                subs[i].columns = ['target' if col == '2' else col for col in subs[i].columns] 

        elif 'third' in args.sub_dir:
                subs[i].columns = ['target' if col == '0' else col for col in subs[i].columns] 
    
        if len(subs[i])> 604:
             subs[i] = subs[i].groupby(df.index // 20).agg({'image_name': 'first', 'y_true': 'first', 'target': 'mean'})
        print(f"len sub {i}:",len(subs[i]))
    if rank_mean:

        sub_probs = [sub.target.rank(pct=True).values for sub in subs]
        """
                image_name    target  percentile
        0     ISIC_0080539  0.015354        1.00
        1   ISIC_0080539_0  0.005722        0.25
        """
        if 'first' in args.sub_dir:     
            wts = [1/18]*18
        elif 'second' in args.sub_dir:   
             wts = [1/3]*3 
        elif 'third' in args.sub_dir:     
             wts = [1/8]*8

        sub_ens = np.sum([wts[i]*sub_probs[i] for i in range(len(wts))],axis=0)       
        df_sub = subs[0]
        df_sub['target'] = sub_ens
        df_sub.to_csv(os.path.join(args.sub_dir, f"sa6_final_sub_prove.csv"),index=False)
    else:
        print(subs[0]['target'][0], subs[1]['target'][0], subs[2]['target'][0])
        concatenated_df = pd.concat(subs, keys=range(len(subs)))
        print(concatenated_df.head())
        print(concatenated_df.shape)
        mean_values = concatenated_df.groupby(level=1).mean()
        mean_values['image_name'] = subs[0]['image_name']
        mean_values.to_csv(f"sa6_final_sub_prove_mean.csv",index=False)
     