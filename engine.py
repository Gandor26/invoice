import torch as tc
from torch import optim
from torch import nn
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from model import AlexBoWNet, AlexNet
from dataset import ClassificationDataset, CombinedDataset
from metric import AverageMeter, TimeMeter, TopKAccuracy
from utils import get_dir, get_logger

class BaseEngine(object):
    def __init__(self, args):
        self.dataset = ClassificationDataset(CombinedDataset, root=args.data_dir, cuda=args.cuda,
                seed=args.seed, stratified=False, threshold=args.threshold)
        self._make_model(args)
        if args.cuda and tc.cuda.is_available():
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
        self.dump_dir = get_dir(args.dump_dir)

    def _make_model(self, args):
        raise NotImplementedError

    def _make_optimizer(self, args):
        raise NotImplementedError

    def _make_loss(self, args):
        raise NotImplementedError

    def _make_metric(self, args):
        raise NotImplementedError

    def dump(self, epoch, tag=None, model=True, optimizer=True, decayer=True):
        logger = get_logger('train.{}'.format(self.__class__.__name__))
        state = {'epoch': epoch}
        if model:
            state['model'] = self.model.state_dict()
        if optimizer:
            state['optimizer'] = self.optimizer.state_dict()
        if decayer and (getattr(self, 'decayer', None) is not None):
            state['decayer'] = self.decayer.state_dict()
        tag = tag or 'default'
        tc.save(state, os.path.join(self.dump_dir, 'state_{}.pkl'.format(tag)))
        logger.info('Checkpoint dumped')

    def load(self, model=True, optimizer=True, decayer=True, tag=None):
        logger = get_logger('train.{}'.format(self.__class__.__name__))
        tag = tag or 'default'
        try:
            state = tc.load(os.path.join(self.dump_dir, 'state_{}.pkl'.format(tag)))
        except FileNotFoundError:
            return 0
        if model and (state.get('model') is not None):
            self.model.load_state_dict(state['model'])
        if optimizer and (state.get('optimizer') is not None):
            self.optimizer.load_state_dict(state['optimizer'])
        if decayer and (state.get('decayer') is not None) and (getattr(self, 'decayer', None) is not None):
            self.decayer.load_state_dict(state['decayer'])
        logger.info('Checkpoint loaded')
        return state['epoch']

    def eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class ImageBoWEngine(BaseEngine):
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

    def eval(self, valid_mode=True):
        dataloader = self.dataset.valid_loader if valid_mode else self.dataset.test_loader
        loss_meter = AverageMeter('valid_loss' if valid_mode else 'test_loss')
        accuracy_meter = AverageMeter('valid_accuracy' if valid_mode else 'test_accuracy')
        for samples, labels in tqdm(dataloader(self.batch_size), desc=('Valid' if valid_mode else 'Test')):
            with tc.no_grad():
                image, bow = samples
                preds = self.model(image, bow)
            loss = self.loss_func(preds, labels)
            loss_meter.add(loss.item(), labels.size(0))
            acc = self.metric(preds, labels)
            accuracy_meter.add(acc.item(), labels.size(0))
        return loss_meter, accuracy_meter

    def train(self, num_epochs, resume=False):
        if resume:
            start_epoch = self.load()
        else:
            start_epoch = 0
        logger = get_logger('train.{}'.format(self.__class__.__name__), clear=(not resume))
        timer = TimeMeter()
        train_meter = AverageMeter('train_loss')
        for epoch in range(start_epoch, start_epoch+num_epochs):
            train_meter.reset()
            for samples, labels in tqdm(self.dataset.train_loader(self.batch_size, self.num_training_samples),
                    desc='Train epoch {}'.format(epoch+1)):
                image, bow = samples
                preds = self.model(image, bow)
                loss = self.loss_func(preds, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_meter.add(loss.item(), labels.size(0))
            vm1, vm2 = self.eval(valid_mode=True)
            self.decayer.step(vm2.read())
            logger.info('Epoch {:02d}, elapsed Time {:.2f}, {} = {:.4f}, {} = {:.4f}, {} = {:.4f}'.format(
                epoch+1, timer.read(), train_meter.tag, train_meter.read(), vm1.tag, vm1.read(), vm2.tag, vm2.read()))

