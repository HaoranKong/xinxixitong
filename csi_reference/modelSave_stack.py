import math
import os

import torch
import torch.nn as nn

class ModelSave_stack:
    def __init__(self, config, stack_size):
        self.config = config
        self.record_txt = open(self.config.workdir + "/result_stack.csv", "w")

        self.stack_size = stack_size
        self.stack_indicator = 0
        self.loss_stack = torch.tensor([math.inf for i in range(self.stack_size)])
        self.epoch_stack = torch.tensor([0 for i in range(self.stack_size)], dtype=torch.int16)
        self.path = None

    def model_save(self, model, save_path, loss):
        [path, suffix] = save_path.split("_EP")
        epoch_num = eval(suffix.split(".")[0])
        self.path = path

        saveIndex = self.decideSaveSatck(loss, epoch_num)
        if saveIndex != None:
            savePath_new = path + ("_checkpoint%d.model" % saveIndex)

            torch.save(model, savePath_new)
            self.recordModel()                              # 讲模型存储信息存到文件当中

    def decideSaveSatck(self, loss, epoch_num):
        index = self.stack_indicator
        if loss < self.loss_stack[index] and loss < self.loss_stack[(index + self.stack_size - 1) % self.stack_size]:
            self.loss_stack[index] = loss
            self.epoch_stack[index] = epoch_num
            self.stack_indicator = (self.stack_indicator + 1) % self.stack_size
            return index
        else:
            return None

    def recordModel(self):
        str = ""
        for i in range(self.stack_size):
            str = str + "checkpoints, %d, epoch, %04d, loss, %2.4f, " % (i, self.epoch_stack[i], self.loss_stack[i])
        min_index = torch.argmin(self.loss_stack)
        str = str + "best_checkpoints, %d, epoch %04d, loss, %2.4f\n" % (min_index, self.epoch_stack[min_index], self.loss_stack[min_index])
        self.record_txt.write(str)
        self.record_txt.flush()

    def read_bestModel(self):
        min_index = torch.argmin(self.loss_stack)
        best_model_path = self.path + ("_checkpoint%d.model" % min_index)
        best_model = torch.load(best_model_path)
        return best_model



