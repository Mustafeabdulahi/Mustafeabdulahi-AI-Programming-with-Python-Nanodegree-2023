import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to category to name mapping json')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def load_checkpoint(file_path='checkpoint.pth'):
    checkpoint = torch.load(file_path)
    arch = checkpoint['arch']
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    for param in model.parameters():
        param.requires_grad = False
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    learn_rate = checkpoint['learning_rate']
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learn_rate)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    return model, optimizer, epochs, learn_rate


def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image)
    return img_tensor

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    if title:
        ax.set_title(title)
    ax.axis('off')
    return ax

def predict(image_path, model, topk=5, device=torch.device("cpu")):
    if device.type == 'cuda':
        model = model.to(device)
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(img_tensor)
    ps = torch.exp(output)
    top_p, top_indices = ps.topk(topk, dim=1)
    top_p = top_p.cpu().numpy().tolist()[0]
    top_indices = top_indices.cpu().numpy().tolist()[0]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    return top_p, top_classes

def display_predicted_img(image_path, probs, classes, model, cat_to_name):
    flower_names = [cat_to_name[str(name)] for name in classes]
    img_tensor = process_image(image_path)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10,5))
    ax1.axis('off')
    ax1.set_title(flower_names[0])
    imshow(img_tensor, ax=ax1)
    ax2.barh(flower_names[::-1], probs[::-1])
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Flower Name')
    ax2.set_title(f'Top {len(probs)} Predicted Flowers')
    plt.tight_layout()
    plt.show()

def main():
    args = get_input_args()
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model, optimizer, epochs, learn_rate = load_checkpoint(file_path=args.checkpoint)
    probs, classes = predict(args.image_path, model, args.top_k, device)
    display_predicted_img(args.image_path, probs, classes, model, cat_to_name)

#Execute this command in the terminal to test the image prediction.
#python predict.py 'flowers/test/28/image_05230.jpg' checkpoint.pth

if __name__ == "__main__":
    main()
