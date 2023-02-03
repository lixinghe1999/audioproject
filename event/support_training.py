from torchvision.models import resnet18, ResNet18_Weights
if __name__ == "__main__":
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)