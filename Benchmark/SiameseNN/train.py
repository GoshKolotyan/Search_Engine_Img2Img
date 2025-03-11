import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  # <-- Progress bar

from losses import ContrastiveLoss
from model import SiameseNetwork
from dataloader import SiameseDataset


def train(
    num_epochs: int = 5,
    model: nn.Module = None,
    criterion: nn.Module = None,
    device: str = "cpu",
    dataloader: DataLoader = None,
    lr: float = 0.001,
    log_dir: str = "runs/siamese_experiment"
):
    """
    Trains a SiameseNetwork model using a contrastive loss.

    Args:
        num_epochs: Number of training epochs.
        model: The model instance to train.
        criterion: The loss function to use, e.g., ContrastiveLoss.
        device: 'cpu' or 'cuda' for training device.
        dataloader: PyTorch DataLoader providing (img_1, img_2, label) batches.
        lr: Learning rate for the Adam optimizer.
        log_dir: Directory where TensorBoard logs will be stored.
    """

    # Instantiate model and criterion if they are not provided
    if model is None:
        model = SiameseNetwork()
    if criterion is None:
        criterion = ContrastiveLoss()

    print("Training starts...")
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for _, (img_1, img_2, label) in enumerate(progress_bar):
            img_1, img_2, label = img_1.to(device), img_2.to(device), label.to(device)

            optimizer.zero_grad()
            embd_1, embd_2 = model(img_1, img_2)

            loss = criterion(embd_1, embd_2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_train_loss:.4f}")

        writer.add_scalar("Train/Loss", avg_train_loss, epoch)

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root_dir = "data"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    siamese_dataset = SiameseDataset(
        root_dir=root_dir,
        transform=transform,
        negative_ratio=1.0,
        seed=42
    )

    dataloader = DataLoader(
        dataset=siamese_dataset,
        batch_size=16,
        num_workers=8,
        shuffle=True,
        pin_memory=True
    )

    train(
        num_epochs=5,
        model=SiameseNetwork(),
        criterion=ContrastiveLoss(),
        device=device,
        dataloader=dataloader,
        lr=0.001,
        log_dir="runs/siamese_experiment"
    )
