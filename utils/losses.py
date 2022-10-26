import torch
from torch import nn


class TeacherLoss(nn.Module):
    def __init__(self):
        super(TeacherLoss, self).__init__()
        self.loss = nn.NLLLoss()

    def forward(self, model_output, batch):
        return self.loss(model_output['log_preds'], batch['labels'])


class StudentLoss(nn.Module):
    def __init__(self,
                 coeff):
        super(StudentLoss, self).__init__()
        assert coeff >= 0 and coeff <=1, "Setup appropriate coeff for StudentLoss"
        self.coeff = nn.Parameter(torch.FloatTensor([coeff]), requires_grad=False)
        self.loss_task = nn.NLLLoss()
        self.loss_distill = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, model_output, batch):
        return self.loss_task(model_output['log_preds'], batch['labels']) * self.coeff + \
               self.loss_distill(model_output['log_preds'], batch['teacher_soft_labels']) * (1 - self.coeff)


