from __future__ import print_function

import argparse
import os
import time
from enum import Enum
#from tensorboardX import SummaryWriter
from torchvision import datasets, transforms, models
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))


#def train(args, model, device, train_loader, optimizer, epoch, writer, criterion, data_counter):
def train(args, model, device, train_loader, optimizer, epoch, criterion, data_counter):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        data_counter['train_size'] += data.size(0)
        
        data, target = data.to(device, non_blocking = True), target.to(device, non_blocking = True)

        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        top5.update(acc5[0], data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))
        #     niter = epoch * len(train_loader) + batch_idx

        #     writer.add_scalar('loss', loss.item(), niter)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            progress.display(batch_idx + 1)
        
#def test(args, model, device, test_loader, writer, epoch, criterion):
def test(args, model, device, val_loader, epoch, criterion, data_counter):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
            #     if args.gpu is not None and torch.cuda.is_available():
            #         images = images.cuda(args.gpu, non_blocking=True)
            #    # if torch.backends.mps.is_available():
            #    #     images = images.to('mps')
            #    #     target = target.to('mps')
            #     if torch.cuda.is_available():
            #         target = target.cuda(args.gpu, non_blocking=True)
                images, target = images.to(device, non_blocking = True), target.to(device, non_blocking = True)
                 
                data_counter['test_size'] += images.size(0)
                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.log_interval == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    model.eval()

    run_validate(val_loader)

    progress.display_summary()

    return top1.avg

    # with torch.no_grad():
    #     for data, target in test_loader:
    #         total_size += data.size(0)
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         test_loss += criterion(output, target).item() # sum up batch loss

    #         pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= total_size
    # print(f'Epoch {epoch} : test_size={total_size}')
    # print('\naccuracy={:.4f}\n'.format(float(correct) * 100 / total_size))
    # print('\n test loss={:.6f}\n'.format(test_loss))
    # print("Test Epoch {} : data_size {} : accuracy {:.4f} : loss {:.6f} ".format(epoch, total_size, float(correct) * 100 / total_size, test_loss))
    # writer.add_scalar('accuracy', float(correct) * 100 / total_size, epoch)


def should_distribute():
    return dist.is_available() and WORLD_SIZE > 1


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Example')
    
    parser.add_argument('--model-name', type=str, default='resenet18', metavar='ARCH')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    # need to check
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--dir', default='logs', metavar='L',
                        help='directory where summary logs are stored')
    if dist.is_available():
        parser.add_argument('--backend', type=str, help='Distributed backend',
                            choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                            default=dist.Backend.GLOO)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print('Using CUDA')

    #writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if should_distribute():
        print('Using distributed PyTorch with {} backend\n'.format(args.backend))
        dist.init_process_group(backend=args.backend)
        print('WORLD_SIZE: {}\n'.format(dist.get_world_size()))
        print('RANK: {}\n'.format(dist.get_rank()))

    traindir = os.path.join('/data', 'train')
    valdir = os.path.join('/data', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=False, 
        sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size, shuffle=False, 
        sampler=val_sampler, **kwargs)

    # user model
    model = models.__dict__[args.model_name]().to(device)
    #model = Net().to(device)

    if is_distributed():
        Distributor = nn.parallel.DistributedDataParallel
        model = Distributor(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    data_counter = {'train_size':0, 'test_size':0}

    before_train = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        train_start = time.time()
        
        train(args, model, device, train_loader, optimizer, epoch, criterion, data_counter)
        
        acc1 = test(args, model, device, test_loader, epoch, criterion, data_counter)

        scheduler.step()
        
        train_end = time.time()
        print("Trainig time for epoch {}: {:.2f} minutes".format(epoch, (train_end-train_start)/60.0))
        print(f"Current data size : {data_counter}")
    
    end_train = time.time()
    print("Total Trainig time for all {} epochs: {:.2f} minutes".format(args.epochs, (end_train-before_train)/60.0))
        
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#     def all_reduce(self):
#         if torch.cuda.is_available():
#             device = torch.device("cuda")
# #        elif torch.backends.mps.is_available():
# #            device = torch.device("mps")
#         else:
#             device = torch.device("cpu")
#         total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
#         dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
#         self.sum, self.count = total.tolist()
#         self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
