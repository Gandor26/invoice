import torch as tc
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
from model import Model
from loader import ImageDataset, SubsetWeightedSampler, SubsetSequentialSampler, split_train_valid
from metric import AverageMeter, TimeMeter
from utils import get_dir

class Engine(object):
    def __init__(self, args):
        tc.set_default_dtype(tc.double)
        tc.initial_seed(args.seed)
        if args.cuda:
            tc.cuda.initial_seed(args.seed)
        self.train_set = ImageDataset(mode='train', cuda=args.cuda, seed=args.seed, margin=args.margin, threshold=args.threshold)
        self.test_set = ImageDataset(mode='test', cuda=args.cuda)
        idx_train, idx_test = split_train_valid(self.train_set, valid_split=args.valid_split)
        self.train_loader = DataLoader(self.train_set, args.batch_size,
                sampler=SubsetWeightedSampler(idx_train, self.train_set.get_weight(idx_train), args.num_training_samples))
        self.valid_loader = DataLoader(self.train_set, args.batch_size,
                sampler=SubsetSequentialSampler(idx_valid))
        self.test_loader = DataLoader(self.test_set, args.batch_size, shuffle=False)

        model = Model(10, args.dropout)
        self.model = model.cuda() if args.cuda else model
        self.optimizer = optim.SGD(self.model.parameters(), args.lr, momentum=args.mmtm, weight_decay=args.wd)
        self.decayer = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.2)
        self.dump_dir = get_dir(args.dump_dir)

    def dump(self, epoch, metric, model=True, optimizer=True, decayer=True, tag=None):
        state = {'epoch': epoch, 'metric': metric}
        if model:
            state['model'] = self.model.state_dict()
        if optimizer:
            state['optimizer'] = self.optimizer.state_dict()
        if decayer:
            state['decayer'] = self.decayer.state_dict()
        tag = tag or 'default'
        tc.save(state, os.path.join(self.dump_dir, 'state_{}.pkl'.format(tag)))
        print('Checkpoint dumped')

    def load(self, model=True, optimizer=True, decayer=True, tag=None):
        tag = tag or 'default'
        state = tc.load(os.path.join(self.dump_dir, 'state_{}.pkl'.format(tag)))
        if model and (state.get('model') is not None):
            self.model.load_state_dict(state['model'])
        if optimizer and (state.get('optimizer') is not None):
            self.optimizer.load_state_dict(state['optimizer'])
        if decayer and (state.get('decayer') is not None):
            self.decayer.load_state_dict(state['decayer'])
        print('Checkpoint loaded')
        return state['epoch'], state['metric']

    def eval(self):
        meter = AverageMeter('acc')
        for samples, labels in tqdm(self.valid_loader, desc='Valid'):
            with tc.no_grad():
                preds = self.model(samples)
            _, top = preds.topk(1, dim=1)
            acc = labels.eq(top.squeeze(dim=1)).float().mean()
            meter.add(acc.item(), labels.size(0))
        return meter.read()

    def train(self, num_epochs, resume=False):
        if resume:
            start_epoch, best_acc = self.load()
        else:
            start_epoch, best_acc = 0, 0
        timer = TimeMeter()
        meter = AverageMeter()
        for epoch in range(start_epoch, num_epochs):
            self.decayer.step()
            for samples, labels in tqdm(self.train_loader, desc='Train epoch {}'.format(epoch+1)):
                preds = self.model(samples)
                loss = F.cross_entropy(preds, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                meter.add(loss.item(), labels.size(0))
            acc = self.eval()
            if acc > best_acc:
                best_acc = acc
                self.dump(epoch+1, acc)
            print('Epoch {:02d}, elapsed Time {:.2f}, loss = {:.4f}, acc = {:.4f}'.format(
                epoch+1, timer.read(), meter.read(), acc))

