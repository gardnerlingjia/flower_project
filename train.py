import argparse
import os
import torch
from torch import nn, optim

from utils import (
    get_data_loaders,
    build_model,
    get_classifier_parameters,
    validation,
    save_checkpoint
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a flower classifier")
    parser.add_argument("data_dir", type=str, help="Path to dataset")
    parser.add_argument("--save_dir", type=str, default="checkpoint.pth",
                        help="Path to save checkpoint")
    parser.add_argument("--arch", type=str, default="vgg13",
                        help="Model architecture: vgg13, vgg16, resnet18")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--hidden_units", type=int, default=512,
                        help="Number of hidden units")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU if available")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset, train_loader, valid_loader, test_loader = get_data_loaders(args.data_dir)

    model = build_model(args.arch, args.hidden_units)
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        get_classifier_parameters(model, args.arch),
        lr=args.learning_rate
    )

    steps = 0
    print_every = 40

    for epoch in range(args.epochs):
        running_loss = 0
        model.train()

        for images, labels in train_loader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss, val_accuracy = validation(model, valid_loader, criterion, device)

                print(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"Train Loss: {running_loss / print_every:.3f} | "
                    f"Valid Loss: {val_loss:.3f} | "
                    f"Valid Accuracy: {val_accuracy:.3f}"
                )

                running_loss = 0
                model.train()

    save_path = args.save_dir

    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, "checkpoint.pth")

    save_checkpoint(
        model=model,
        train_dataset=train_dataset,
        save_path=save_path,
        arch=args.arch,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    print(f"Checkpoint saved to: {save_path}")


if __name__ == "__main__":
    main()