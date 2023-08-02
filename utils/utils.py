import torch
import torch.nn as nn


class Aggregation_Separation_Loss(nn.Module):
    def __init__(self, distance=nn.SmoothL1Loss(), constraint=10):
        super().__init__()
        self.distance = distance
        self.constraint = constraint
        self.cosine = isinstance(self.distance, nn.CosineSimilarity)

    def forward(self, distributions, labels):
        device = distributions.device
        loss_inner = torch.tensor(0).to(device)
        loss_outer = torch.tensor(0).to(device)
        n1 = 0
        n2 = 0
        if self.cosine:
            for i, s1 in enumerate(distributions):
                for j, s2 in enumerate(distributions):
                    if labels[i].item() == labels[j].item():
                        loss_inner = loss_inner + (-1 * self.distance(
                            s1.unsqueeze(0), s2.unsqueeze(0)) + 1)
                        n1 += 1
                    else:
                        loss_outer = loss_outer + (-1 * self.distance(
                            s1.unsqueeze(0), s2.unsqueeze(0)) + 1)
                        n2 += 1
        else:
            for i, s1 in enumerate(distributions):
                for j, s2 in enumerate(distributions):
                    if labels[i].item() == labels[j].item():
                        loss_inner = loss_inner + self.distance(s1, s2)
                        n1 += 1
                    else:
                        loss_outer = loss_outer + self.distance(s1, s2)
                        n2 += 1

        loss_penalty = ((distributions.norm(2, dim=1) -
                         self.constraint)**2).mean()

        if n1 != 0:
            loss_inner = loss_inner / n1
        if n2 != 0:
            loss_outer = loss_outer / n2

        return loss_inner, loss_outer, loss_penalty


class Aggregation_Separation_Loss_with_multi_distribution(
        Aggregation_Separation_Loss):
    def __init__(self, distance=nn.SmoothL1Loss(), constraint=10):
        super().__init__(distance, constraint)

    def forward(self, distributions, labels, lambda_distribution):
        device = distributions[0].device
        loss_inner = torch.tensor(0).to(device)
        loss_outer = torch.tensor(0).to(device)
        loss_penalty = torch.tensor(0).to(device)
        for idx, distribution in enumerate(distributions):
            _loss_inner, _loss_outer, _loss_penalty = super().forward(
                distribution, labels)
            loss_inner = loss_inner + lambda_distribution[idx] * _loss_inner
            loss_outer = loss_outer + lambda_distribution[idx] * _loss_outer
            loss_penalty = loss_penalty + lambda_distribution[
                idx] * _loss_penalty
        return loss_inner, loss_outer, loss_penalty


def make_report(loss_average,
                accuracy,
                sensitivity,
                specificity,
                Youden_Index,
                epoch=None,
                stage="train"):
    if epoch is not None:
        log = "epoch: %s \n" % epoch
    else:
        log = ""
    if stage == "train":
        log += "      \t loss     \t accuracy \t sensitivity \t specificity \t Youden Index \n"
    log += "%s:\t %.5f \t %.5f \t %.5f \t\t %.5f \t\t %.5f \n" \
        % (stage, loss_average, accuracy, sensitivity, specificity, Youden_Index)
    return log


def make_report2(information, epoch=None, stage="train"):
    log = ""
    if stage == "train":
        log += "epoch: %s \n" % epoch
        log += "    train:\n    "
    else:
        log += "    test:\n    "

    for k, v in information.items():
        log += "%s:%.5f    " % (k, v)

    log += "\n"
    return log


def write_log(path, log):
    with open(path, "a") as f:
        f.write(log)
