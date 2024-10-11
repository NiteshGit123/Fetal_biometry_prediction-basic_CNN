import torch
from torch import optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.1, best_model_path='best_model_landmark.pth'):
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for images, biometry_points in tqdm(train_loader):
            images, biometry_points = images.to(device), biometry_points.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.float(), biometry_points.float())
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss)

        # Validation step
        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0
            for images, biometry_points in val_loader:
                images, biometry_points = images.to(device), biometry_points.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, biometry_points.float())
                epoch_val_loss += val_loss.item()
            val_losses.append(epoch_val_loss)

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}')

    return train_losses, val_losses
