class s6_train_loss():
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
            optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(
                desc=f'Loss={train_loss.item()} Batch_id={batch_idx} Train Accuracy={100 * correct / processed:0.2f}')

            train_loss /= processed
            self.stats.append_loss(round(train_loss.item(), 6))

        train_acc = round((100. * correct / len(train_loader.dataset)), 2)
        self.stats.append_acc(train_acc)

        print(
            f'Train set: Epoch : {current_epoch}/{self.total_epochs} Average loss: {train_loss :.4f}, Train Accuracy: {train_acc}')