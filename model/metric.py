import torch
import pdb


def accuracy(output, target, ignore_index=None):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        if ignore_index is None:
            total_valid = float(len(target))
            correct += torch.sum((pred == target).float())
        else:
            total_valid = torch.sum((target != ignore_index).float())
            correct += torch.sum(((pred == target) * (target != ignore_index)).float())
    return correct / total_valid


def top_k_acc(output, target, k=3, ignore_index=None):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        if ignore_index is None:
            total_valid = float(len(target))
            correct += torch.sum((pred == target[:,None]).float())
        else:
            total_valid = torch.sum((target != ignore_index).float())
            correct += torch.sum(((pred == target[:,None]) * (target != ignore_index)[:,None]).float())
    return correct / total_valid

def mean_rank(output, target):
    with torch.no_grad():
        sorted_output = torch.argsort(output, dim=-1, descending=True)
        sorted_result = sorted_output == target[:, None]
        ranks = torch.nonzero(sorted_result)[:, 1]
        mr = torch.mean(ranks.float())
    return mr

def mean_average_precision(output, relevance_labels):
    with torch.no_grad():
        sorted_output = torch.argsort(output, dim=-1, descending=True)
        sorted_labels = torch.gather(relevance_labels, -1, sorted_output).float()
        cum_correct = torch.cumsum(sorted_labels, dim=-1)
        cum_precision = cum_correct / torch.arange(start=1,end=cum_correct.shape[-1]+1, device=cum_correct.device)[None, :]
        cum_precision = cum_precision * sorted_labels
        total_valid = torch.sum(sorted_labels, dim=-1)
        mean_average_precision = torch.mean(torch.sum(cum_precision, dim=-1)/total_valid)

    return mean_average_precision

def average_precision(output, relevance_labels):
    with torch.no_grad():
        sorted_output = torch.argsort(output, dim=-1, descending=True)
        sorted_labels = torch.gather(relevance_labels, -1, sorted_output).float()
        cum_correct = torch.cumsum(sorted_labels, dim=-1)
        cum_precision = cum_correct / torch.arange(start=1,end=cum_correct.shape[-1]+1, device=cum_correct.device)[None, :]
        cum_precision = cum_precision * sorted_labels
        total_valid = torch.sum(sorted_labels, dim=-1)
        total_valid[total_valid==0] = 1
        average_precision = torch.sum(cum_precision, dim=-1)/total_valid
    return average_precision