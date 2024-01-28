import torch
import torch.nn as nn
import torch.nn.functional as F

class Soft_Distillation_Loss(nn.Module):
    def __init__(self, lambda_balancing: float):
        super(Soft_Distillation_Loss, self).__init__()

        self.lambda_balancing = lambda_balancing
        self.CE_student = nn.MSELoss()
        self.CE_Teacher = nn.MSELoss()
        self.KLD_teacher = nn.KLDivLoss(reduction='batchmean')

    def forward(self, teacher_y, student_y, y,temperature):
        loss = ((1 - self.lambda_balancing) * (self.CE_student(student_y, y)) + (
                    self.lambda_balancing * (temperature ** 3) *
                    self.KLD_teacher(student_y / temperature, teacher_y / temperature)))
        return loss

class Hard_Distillation_Loss(nn.Module):
    def __init__(self):
        super(Hard_Distillation_Loss, self).__init__()

        self.CE_teacher = nn.CrossEntropyLoss()
        self.CE_student = nn.CrossEntropyLoss()

    def forward(self, teacher_y, student_y, y):
        loss = (0.5) * (self.CE_student(student_y, y)) + \
               (0.2) * (self.CE_teacher(teacher_y, y)) \
               + (0.3) * (self.CE_student(student_y, teacher_y))

        return loss
