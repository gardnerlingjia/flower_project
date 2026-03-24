import argparse
import torch

from utils import load_checkpoint, predict, load_category_names


def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument("input", type=str, help="Path to input image")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default=None,
                        help="Path to category names JSON file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = load_checkpoint(args.checkpoint)

    probs, classes = predict(args.input, model, topk=args.top_k, device=device)

    if args.category_names:
        cat_to_name = load_category_names(args.category_names)
        names = [cat_to_name.get(cls, cls) for cls in classes]
    else:
        names = classes

    print("\nTop predictions:")
    for i, (name, prob, cls) in enumerate(zip(names, probs, classes), start=1):
        print(f"{i}. {name} (class {cls}) - probability: {prob:.4f}")


if __name__ == "__main__":
    main()