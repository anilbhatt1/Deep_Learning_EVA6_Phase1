from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

class train_losses():
    def __init__(self, model, device, train_loader, train_stats, optimizer, total_epochs):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.stats = train_stats
        self.optimizer = optimizer
        self.total_epochs = total_epochs

    def s6_train(self, current_epoch, L1_factor=0):
        self.model.train()
        pbar = tqdm(self.train_loader)
        train_loss, correct, processed = 0, 0, 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(data)
            train_loss = F.nll_loss(y_pred, target)

            # Updating train loss with L1 loss
            L1_Crit = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
            reg_loss = 0
            for param in self.model.parameters():
                zero_vector = torch.rand_like(param) * 0
                reg_loss += L1_Crit(param, zero_vector)
            train_loss += L1_factor * reg_loss

            train_loss.backward()
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={train_loss.item()} Batch_id={batch_idx} Train Accuracy={100 * correct / processed:0.2f}')

            train_loss /= processed
            self.stats.append_loss(round(train_loss.item(), 6))

        train_acc = round((100. * correct / len(self.train_loader.dataset)), 2)
        self.stats.append_acc(train_acc)

        print(
            f'Train set: Epoch : {current_epoch}/{self.total_epochs} Average loss: {train_loss :.4f}, Train Accuracy: {train_acc}')

    def s7_train(self, current_epoch, L1_factor=0):
        self.model.train()
        pbar = tqdm(self.train_loader)
        train_loss, correct, processed = 0, 0, 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(data)
            train_loss = F.nll_loss(y_pred, target)

            # Updating train loss with L1 loss
            L1_Crit = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
            reg_loss = 0
            for param in self.model.parameters():
                zero_vector = torch.rand_like(param) * 0
                reg_loss += L1_Crit(param, zero_vector)
            train_loss += L1_factor * reg_loss

            train_loss.backward()
            self.optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={train_loss.item()} Batch_id={batch_idx} Train Accuracy={100 * correct / processed:0.2f}')

            train_loss /= processed
            self.stats(round(train_loss.item(), 6), 'train_loss')

        train_acc = round((100. * correct / len(self.train_loader.dataset)), 2)
        self.stats(train_acc, 'train_acc')

        print(
            f'Train set: Epoch : {current_epoch}/{self.total_epochs} Average loss: {train_loss :.4f}, Train Accuracy: {train_acc}')

    def s8_train(self, current_epoch, scheduler, tb_writer, L1_factor=0):
        self.model.train()
        pbar = tqdm(self.train_loader)
        train_loss, correct, processed = 0, 0, 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            y_pred = self.model(data)
            train_loss = F.nll_loss(y_pred, target)

            # Updating train loss with L1 loss
            L1_Crit = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
            reg_loss = 0
            for param in self.model.parameters():
                zero_vector = torch.rand_like(param) * 0
                reg_loss += L1_Crit(param, zero_vector)
            train_loss += L1_factor * reg_loss

            train_loss.backward()
            self.optimizer.step()

            lr = 0
            if scheduler and not (isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)):
                lr = scheduler.get_last_lr()[0]  # Won't work for ReduceLRonPlateau
            else:
                lr = self.optimizer.param_groups[0]['lr']

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={train_loss.item()} Batch_id={batch_idx} Train Accuracy={100 * correct / processed:0.2f}')

            tb_writer.add_scalar('Train loss', round(train_loss.item(), 4), global_step=current_epoch)

            self.stats(round(train_loss.item(), 6), 'train_loss')

        train_acc = round((100. * correct / len(self.train_loader.dataset)), 2)
        self.stats(train_acc, 'train_acc')
        tb_writer.add_scalar('Acc/Train', train_acc, global_step=current_epoch)

        print(
            f'Train set: Epoch : {current_epoch}/{self.total_epochs} Average loss: {train_loss :.4f}, Train Accuracy: {train_acc}')