import torch
import pandas as pd
from model import BiometryDetection
from data_loader import get_dataloaders
from train import train_model
from test import evaluate_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiometryDetection()
    model = model.to(device)

    labels_df = pd.read_csv('/kaggle/input/landmark-data/correct_data.csv')
    train_loader, val_loader, test_loader = get_dataloaders(labels_df)

    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    evaluate_model(model, test_loader, device)

if __name__ == '__main__':
    main()
