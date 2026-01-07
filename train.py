import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightly.loss import NTXentLoss

# Import the logic you just built in the other files
from models import get_4channel_resnet, SimCLRModel
from data_utils import SatellitePatchDataset, SatelliteTransform


def start_training(patch_dir, save_path, epochs=20, batch_size=256, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on device: {device}")

    # 1. Prepare Data
    transform = SatelliteTransform(input_size=64)
    dataset = SatellitePatchDataset(patch_dir=patch_dir, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    # 2. Initialize Model
    backbone = get_4channel_resnet(pretrained=True)
    model = SimCLRModel(backbone).to(device)

    # 3. Define Loss and Optimizer
    criterion = NTXentLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # Batch contains (view1, view2) from SatelliteTransform
            x0, x1 = [x.to(device) for x in batch]

            # Forward pass
            z0, z1 = model(x0), model(x1)
            loss = criterion(z0, z1)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    # 5. Save ONLY the backbone (the part we need for analysis)
    torch.save(model.backbone.state_dict(), save_path)
    print(f"✅ Training Complete. Backbone saved to: {save_path}")


if __name__ == "__main__":
    # Example local paths (In Kaggle, you'd update these)
    PATCH_DIRECTORY = "./train_patches"
    MODEL_SAVE_PATH = "./final_backbone.pth"

    start_training(PATCH_DIRECTORY, MODEL_SAVE_PATH)
