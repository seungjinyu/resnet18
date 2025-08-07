import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_cus_def import ResNetStem

def print_backward_graph(fn, depth=0):
    print("  " * depth + f"{fn}")
    if hasattr(fn, "next_functions"):
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                print_backward_graph(next_fn, depth + 1)


transform = transforms.Compose([
    transforms.Resize(224),     # ResNet은 원래 224x224 입력을 기대함
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

torch.manual_seed(44)

images, labels = next(iter(loader))  # 하나 꺼냄
print(f"Original image shape: {images.shape}")

stem = ResNetStem()
# for name,param in stem.named_parameters():
#     print(f"{name} ( shape: {param.shape})")
#     flat = param.view(-1)
#     preview = " ".join(f"{v.item():.4f}" for v in flat[:5])
#     print(f"{name} (shape: {param.shape}): {preview} ...")
#     # for i , v in enumerate(flat):
#     #     print(f". [{i}]: {v.item():.4f}")
#     print()
# stem.eval()
out = stem(images)

out = torch.flatten(out,1)
in_features = out.shape[1]   # 25088
fc = nn.Linear(in_features,10)

logits = fc(out)
labels = torch.tensor([3])
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
loss.backward()

for name, param in stem.named_parameters():
    if param.grad is not None:
        flat_grad = param.grad.view(-1)
        preview = " ".join(f"{v.item():.4f}" for v in flat_grad[:5])
        print(f"{name} grad (shape: {param.grad.shape}): {preview} ...")
    else:
        print(f"{name} grad is None")

# print_backward_graph(out.grad_fn)
