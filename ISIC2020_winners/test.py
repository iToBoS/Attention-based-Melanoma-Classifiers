import os
import torch
import geffnet
from PIL import Image
from dataset_multi_output import get_df, get_transforms, MelanomaDataset


#####import df test here

df, df_test, meta_features, n_meta_features, mel_idx,df_synth = get_df(
        'tf_efficientnet_b6',8,'/home/falcon/sana/scratch/Classifier/data',
        512,False,ISIC2020_test=False)
transforms_train, transforms_val = get_transforms(384)
dataset_test = MelanomaDataset(df_test, 'test', meta_features, transform=transforms_val) 
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=64, num_workers=2)
# Set your data directories
weights_dir = "path/to/weights"  # Update with your actual path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = geffnet.create_model('tf_efficientnet_b6', pretrained=False)
model.to(device)

# Loop through each fold and load the corresponding weights
for fold in range(1, 6):
    weights_path = os.path.join(weights_dir, f"fold_{fold}_weights.pth")
    model.load_state_dict(torch.load(weights_path))
    
    # Set the model to evaluation mode
    model.eval()


    # Make predictions on the test set
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())

    # Print or save the predictions for this fold
    print(f"Fold {fold} predictions: {predictions}")

# Now you have predictions for each fold.
# You can perform further analysis or ensemble the predictions as needed.
