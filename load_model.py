#from utils import compute_loss_and_accuracy
#from dataloaders import load_cifar10
import torch

model = torch.load("/home/shomeb/h/henrivas/Documents/A2_datasyn/ConvNet-CIFAR/model_res.pt")
model.eval()
#_, dataloader_val, _ = load_cifar10(32)

#loss, acc = compute_loss_and_accuracy(dataloader_val, model, nn.CrossEntropyLoss())

#print("Loss:", loss)
#print("Acc:", acc)
