import torch as tc
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm
from model import AlexBoWNet as Model
from dataset import ClassificationDataset, CombinedDataset
from metric import AverageMeter, TimeMeter
from utils import get_dir

class Engine(object):
    def __init__(self, args):
        tc.set_default_dtype(tc.double)
        tc.manual_seed(args.seed)
        if args.cuda:
            tc.cuda.manual_seed(args.seed)
        self.dataset = ClassificationDataset(CombinedDataset, root=args.data_dir, cuda=args.cuda,
                seed=args.seed, stratified=False, threshold=args.threshold)
        self.batch_size = args.batch_size
        self.num_training_samples = args.num_training_samples
        model = Model(self.dataset.num_classes, self.dataset.vocab_size, args.dropout)
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
        for samples, labels in tqdm(self.dataset.valid_loader(self.batch_size), desc='Valid'):
            with tc.no_grad():
                image, bow = samples
                preds = self.model(image, bow)
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
            for samples, labels in tqdm(self.dataset.train_loader(self.batch_size, self.num_training_samples), desc='Train epoch {}'.format(epoch+1)):
                image, bow = samples
                preds = self.model(image, bow)
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

