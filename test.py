import torch

def evaluate_model(model, test_loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    test_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, biometry_points in test_loader:
            images, biometry_points = images.to(device), biometry_points.to(device)
            outputs = model(images)
            loss = criterion(outputs, biometry_points.float())
            test_loss += loss.item()
            num_samples += len(images)

    print(f'Test Loss: {test_loss}')
