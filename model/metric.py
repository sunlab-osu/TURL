import torch
import pdb


def accuracy(output, target, ignore_index=None):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        if ignore_index is None:
            total_valid = len(target)
            correct += torch.sum(pred == target).item()
        else:
            total_valid = torch.sum(target != ignore_index)
            correct += torch.sum((pred == target) * (target != ignore_index)).item()
    return correct / total_valid


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
