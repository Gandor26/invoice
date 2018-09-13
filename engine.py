import os
import torch as tc
from torch import optim
from torch import nn
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from model import AlexBoWNet, AlexNet
from dataset import ClassificationDataset, CombinedDataset, CombinedDataset_v2
from metric import AverageMeter, TimeMeter, TopKAccuracy
from utils import get_dir, get_logger

class BaseEngine(object):
    def __init__(self, args):
        self._make_dataset(args)
        self._make_model(args)
        tc.manual_seed(args.seed)
        if args.cuda and tc.cuda.is_available():
            tc.cuda.manual_seed_all(args.seed)
            if tc.cuda.device_count() > 1:
                self.batch_size = args.batch_size * tc.cuda.device_count()
                self.model = DataParallel(self.model)
            else:
                self.batch_size = args.batch_size
                self.model = self.model.cuda()
        else:
            self.batch_size = args.batch_size
        self._make_optimizer(args)
        self._make_loss(args)
        self._make_metric(args)
        self.num_training_samples = args.num_training_samples
        self.tag = args.tag or 'default'
        self.dump_dir = get_dir(args.dump_dir)
        self.logger = get_logger('train.{}'.format(self.__class__.__name__))

    def _make_dataset(self, args):
        raise NotImplementedError

    def _make_model(self, args):
        raise NotImplementedError

    def _make_optimizer(self, args):
        raise NotImplementedError

    def _make_loss(self, args):
        raise NotImplementedError

    def _make_metric(self, args):
        raise NotImplementedError

    def dump(self, epoch, model=True, optimizer=True, decayer=True):
        state = {'epoch': epoch}
        if model:
            state['model'] = self.model.state_dict()
        if optimizer:
            state['optimizer'] = self.optimizer.state_dict()
        if decayer and (getattr(self, 'decayer', None) is not None):
            state['decayer'] = self.decayer.state_dict()
        tc.save(state, os.path.join(self.dump_dir, 'state_{}.pkl'.format(self.tag)))
        self.logger.info('Checkpoint dumped')

    def load(self, model=True, optimizer=True, decayer=True):
        try:
            state = tc.load(os.path.join(self.dump_dir, 'state_{}.pkl'.format(self.tag)))
        except FileNotFoundError:
            return 0
        if model and (state.get('model') is not None):
            self.model.load_state_dict(state['model'])
        if optimizer and (state.get('optimizer') is not None):
            self.optimizer.load_state_dict(state['optimizer'])
        if decayer and (state.get('decayer') is not None) and (getattr(self, 'decayer', None) is not None):
            self.decayer.load_state_dict(state['decayer'])
        self.logger.info('Checkpoint loaded')
        return state['epoch']

    def eval(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def train(self, num_epochs, resume=False):
        raise NotImplementedError


class ImageBoWEngine(BaseEngine):
    def _make_dataset(self, args):
        self.dataset = ClassificationDataset(CombinedDataset, root=args.data_dir, cuda=args.cuda,
                seed=args.seed, stratified=False, threshold=args.threshold)

    def _make_model(self, args):
        self.model = AlexBoWNet(self.dataset.num_classes, self.dataset.vocab_size, args.dropout)

    def _make_optimizer(self, args):
        self.optimizer = optim.SGD([
            dict(params=self.model.features.parameters(), lr=args.lr, momentum=args.mmtm),
            dict(params=self.model.classifier.parameters(), lr=args.lr*10, momentum=0),
            dict(params=self.model.bow.parameters(), lr=args.lr*10, momentum=args.mmtm)])
        self.decayer = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1,
                patience=3, verbose=True, cooldown=1)

    def _make_loss(self, args):
        raw_loss = nn.CrossEntropyLoss()
        def real_loss(preds, labels):
            return raw_loss(preds, labels) + args.L1_decay*sum(p.norm(1) for p in self.model.bow.parameters())
        self.loss_func = real_loss

    def _make_metric(self, args):
        self.metric = TopKAccuracy(top_k=args.top_k)

    def eval(self):
        loss_meter = AverageMeter('valid_loss')
        acc_meter = AverageMeter('valid_accuracy')
        for samples, labels in tqdm(self.dataset.valid_loader(self.batch_size), desc='Validation'):
            with tc.no_grad():
                image, bow = samples
                preds = self.model(image, bow)
            loss = self.loss_func(preds, labels)
            loss_meter.add(loss.item(), labels.size(0))
            acc = self.metric(preds, labels)
            acc_meter.add(acc.item(), labels.size(0))
        return loss_meter, acc_meter

    def test(self):
        epoch = self.load()
        acc_meter = AverageMeter('test_accuracy')
        for samples, labels in tqdm(self.dataset.test_loader(self.batch_size), desc='Test'):
            with tc.no_grad():
                image, bow = samples
                preds = self.model(image, bow)
            acc = self.metric(preds, labels)
            acc_meter.add(acc.item(), labels.size(0))
        self.logger.info('After {} of training, {} = {:.4f}'.format(epoch+1, acc_meter.tag, acc_meter.read()))

    def train(self, num_epochs, resume=False):
        if resume:
            start_epoch = self.load()
        else:
            start_epoch = 0
        timer = TimeMeter()
        tm = AverageMeter('train_loss')
        for epoch in range(start_epoch, start_epoch+num_epochs):
            tm.reset()
            for samples, labels in tqdm(self.dataset.train_loader(self.batch_size, self.num_training_samples),
                    desc='Train epoch {}'.format(epoch+1)):
                image, bow = samples
                preds = self.model(image, bow)
                loss = self.loss_func(preds, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tm.add(loss.item(), labels.size(0))
            vm_loss, vm_acc = self.eval()
            if self.decayer.is_better(vm_acc.read(), self.decayer.best):
                self.dump(epoch)
            self.decayer.step(vm_acc.read())
            self.logger.info('Epoch {:02d}, elapsed Time {:.2f}, {} = {:.4f}, {} = {:.4f}, {} = {:.4f}'.format(
                epoch+1, timer.read(), tm.tag, tm.read(), vm_loss.tag, vm_loss.read(), vm_acc.tag, vm_acc.read()))


class ImageBoWEngine_v2(ImageBoWEngine):
    def _make_dataset(self, args):
        self.dataset = ClassificationDataset(CombinedDataset_v2, root=args.data_dir, cuda=args.cuda,
                seed=args.seed, stratified=False, threshold=args.threshold)

    def eval(self):
        loss_meter = AverageMeter('valid_loss')
        acc_meter = AverageMeter('valid_accuracy')
        for samples, labels in tqdm(self.dataset.valid_loader(self.batch_size), desc='Validation'):
            with tc.no_grad():
                _, image, bow = samples
                preds = self.model(image, bow)
            loss = self.loss_func(preds, labels)
            loss_meter.add(loss.item(), labels.size(0))
            acc = self.metric(preds, labels)
            acc_meter.add(acc.item(), labels.size(0))
        return loss_meter, acc_meter

    def test(self):
        epoch = self.load()
        acc_meter = AverageMeter('test_accuracy')
        test_logger = get_logger('test')
        for samples, labels in tqdm(self.dataset.test_loader(self.batch_size), desc='Test'):
            with tc.no_grad():
                guids, image, bow = samples
                preds = self.model(image, bow)
            acc = self.metric(preds, labels)
            acc_meter.add(acc.item(), labels.size(0))
            _, top = preds.topk(self.metric.top_k, dim=1)
            match = labels.unsqueeze(dim=1).eq(top).any(dim=1, keepdim=False)
            for i, eq in enumerate(match):
                if not bool(eq.item()):
                    guid = guids[i]
                    label = self.dataset.labels.inverse_transform(labels[i].item())
                    preds = self.dataset.labels.inverse_transform(top[i].cpu().numpy())
                    test_logger.info('Prediction failure: {} belongs to {}, but was predicted as {}'.format(guid, label, preds))
        self.logger.info('After {} of training, {} = {:.4f}'.format(epoch+1, acc_meter.tag, acc_meter.read()))

    def train(self, num_epochs, resume=False):
        if resume:
            start_epoch = self.load()
        else:
            start_epoch = 0
        timer = TimeMeter()
        tm = AverageMeter('train_loss')
        for epoch in range(start_epoch, start_epoch+num_epochs):
            tm.reset()
            for samples, labels in tqdm(self.dataset.train_loader(self.batch_size, self.num_training_samples),
                    desc='Train epoch {}'.format(epoch+1)):
                _, image, bow = samples
                preds = self.model(image, bow)
                loss = self.loss_func(preds, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tm.add(loss.item(), labels.size(0))
            vm_loss, vm_acc = self.eval()
            if self.decayer.is_better(vm_acc.read(), self.decayer.best):
                self.dump(epoch)
            self.decayer.step(vm_acc.read())
            self.logger.info('Epoch {:02d}, elapsed Time {:.2f}, {} = {:.4f}, {} = {:.4f}, {} = {:.4f}'.format(
                epoch+1, timer.read(), tm.tag, tm.read(), vm_loss.tag, vm_loss.read(), vm_acc.tag, vm_acc.read()))
