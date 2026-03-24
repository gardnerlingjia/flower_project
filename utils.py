import json
from collections import OrderedDict
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


def get_data_loaders(data_dir):
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"
    test_dir = f"{data_dir}/test"

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    valid_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return train_dataset, train_loader, valid_loader, test_loader


def build_model(arch="vgg13", hidden_units=512):
    if arch == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
        input_features = model.classifier[0].in_features
    elif arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        input_features = model.classifier[0].in_features
    elif arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        input_features = model.fc.in_features
    else:
        raise ValueError("Supported architectures: vgg13, vgg16, resnet18")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(input_features, hidden_units)),
        ("relu", nn.ReLU()),
        ("dropout", nn.Dropout(0.2)),
        ("fc2", nn.Linear(hidden_units, 102)),
        ("output", nn.LogSoftmax(dim=1))
    ]))

    if arch.startswith("vgg"):
        model.classifier = classifier
    elif arch == "resnet18":
        model.fc = classifier

    return model


def get_classifier_parameters(model, arch):
    if arch.startswith("vgg"):
        return model.classifier.parameters()
    return model.fc.parameters()


def validation(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            val_loss += batch_loss.item()

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return val_loss / len(dataloader), accuracy / len(dataloader)


def save_checkpoint(model, train_dataset, save_path, arch, hidden_units, epochs, learning_rate):
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "class_to_idx": train_dataset.class_to_idx,
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")

    model = build_model(
        arch=checkpoint["arch"],
        hidden_units=checkpoint["hidden_units"]
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def process_image(image_path):
    image = Image.open(image_path).convert("RGB")

    resize_size = 256
    crop_size = 224

    width, height = image.size
    if width < height:
        new_width = resize_size
        new_height = int(resize_size * height / width)
    else:
        new_height = resize_size
        new_width = int(resize_size * width / height)

    image = image.resize((new_width, new_height))

    left = (new_width - crop_size) / 2
    top = (new_height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))

    np_image = np.array(image) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path, model, topk=5, device="cpu"):
    model.to(device)
    model.eval()

    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.exp(output)
        top_probs, top_indices = probabilities.topk(topk, dim=1)

    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes


def load_category_names(json_path):
    with open(json_path, "r") as f:
        return json.load(f)