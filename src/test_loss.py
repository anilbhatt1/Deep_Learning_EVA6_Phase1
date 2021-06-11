import torch
import torch.nn as nn
import torch.nn.functional as F

class test_losses():
  def __init__(self, model, device, test_loader, test_stats, total_epochs):
      self.model       = model
      self.device      = device
      self.test_loader = test_loader
      self.stats       = test_stats
      self.total_epochs = total_epochs

  def s6_test(self, current_epoch):
      self.model.eval()
      test_loss, correct, count_wrong = 0, 0, 0
      with torch.no_grad():
          for data, target in self.test_loader:
              data, target = data.to(self.device), target.to(self.device)
              output = self.model(data)
              test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()

              if current_epoch == (self.total_epochs - 1):
                  compare = pred.eq(target.view_as(pred))
                  misclass_idx = (compare == False).nonzero(as_tuple=True)[0].tolist()
                  for i in misclass_idx:
                      self.stats.append_img(data[i])
                      self.stats.append_pred(pred[i].item())
                      self.stats.append_label(target[i].item())

      test_loss /= len(self.test_loader.dataset)
      self.stats.append_loss(round(test_loss,6))
      test_acc = round((100. * correct / len(test_loader.dataset)), 2)
      self.stats.append_acc(test_acc)

      print(f'Test set: Epoch : {current_epoch}/{self.total_epochs} Average loss: {test_loss :.4f}, Test Accuracy: {test_acc}')