import argparse
import logging
import os
import time
from collections import OrderedDict
from datetime import datetime
import yaml

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils

from code.utils import setup_default_logging, random_seed, update_summary, AverageMeter, dispatch_clip_grad, accuracy, get_outdir
from code.models import create_model
from code.optim import create_optimizer_v2, optimizer_kwargs


DATA_MEAN = (0.485, 0.456, 0.406)
DATA_STD = (0.229, 0.224, 0.225)


_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('--data-dir', metavar='DIR', help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='', help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train', help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation', help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False)

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='vit', type=str, metavar='MODEL')
# group.add_argument('--in-channels', type=int, default=3, metavar='N')
group.add_argument('--num-classes', type=int, default=100, metavar='N')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N')
group.add_argument('--img-size', type=int, default=224)
group.add_argument('--input-size', default=(3, 224, 224), nargs=3, type=int, metavar='N N N')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER')
group.add_argument('--momentum', type=float, default=0.9, metavar='M')
group.add_argument('--weight-decay', type=float, default=2e-5)
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM')
group.add_argument('--clip-mode', type=str, default='norm', help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER')
group.add_argument('--sched-on-updates', action='store_true', default=False, help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--epochs', type=int, default=50, metavar='N')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N', help='how many training processes to use (default: 4)')
group.add_argument('--output', default='', type=str, metavar='PATH', help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME', help='name of train experiment, name of sub-folder for output')
group.add_argument('--amp', action='store_true', default=False)
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC', help='Best metric (default: "top1"')
group.add_argument('--log-interval', type=int, default=50, metavar='N')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    print()
    print(args)

    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device=device)
    _logger.info(f'Training with a single process on 1 device ({device}).')

    random_seed(args.seed)

    # create model
    model = create_model(
        args.model,
        img_size=args.img_size,
        in_channels=args.input_size[0],
        num_classes=args.num_classes,
    )
    model.to(device=device)
    # print(model)

    # optimizer
    optimizer = create_optimizer_v2(model.parameters(), **optimizer_kwargs(cfg=args))
    # print(optimizer)

    # data augmentation
    transform_train = transforms.Compose([
        transforms.Resize(args.input_size[-2:]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(DATA_MEAN), std=torch.tensor(DATA_STD)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(args.input_size[-2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor(DATA_MEAN), std=torch.tensor(DATA_STD)),
    ])

    # create the train and eval datasets
    dataset_train = torchvision.datasets.CIFAR100(root='./dataset', train=True, download=args.dataset_download, transform=transform_train)
    dataset_eval = torchvision.datasets.CIFAR100(root='./dataset', train=False, download=args.dataset_download, transform=transform_test)

    # create data loaders w/ augmentation pipeiine
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # loss function
    loss_fn = nn.CrossEntropyLoss().to(device=device)

    # metric
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    output_dir = None
    exp_name = '-'.join([datetime.now().strftime("%Y%m%d-%H%M%S"), args.model])
    output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
    with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
        f.write(args_text)

    # setup learning rate schedule and starting epoch
    lr_scheduler = None
    # lr_scheduler, num_epochs = create_scheduler_v2(
    #     optimizer, 
    #     **scheduler_kwargs(args), 
    #     updates_per_epoch=len(loader_train)
    # )
    # _logger.info(f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    # training
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            epoch,
            model,
            loader_train,
            optimizer,
            loss_fn,
            args,
            device,
            lr_scheduler=lr_scheduler,
        )

        eval_metrics = validate(
            model,
            loader_eval,
            loss_fn,
            args,
            device
        )

        if output_dir is not None:
            lrs = [param_group['lr'] for param_group in optimizer.param_groups]
            update_summary(
                epoch,
                train_metrics,
                eval_metrics,
                filename=os.path.join(output_dir, 'summary.csv'),
                lr=sum(lrs) / len(lrs),
                write_header=best_metric is None,
            )

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])
        

def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device,
        lr_scheduler=None,
):
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    model.train()

    end = time.time()
    num_batches_per_epoch = len(loader)
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch

    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        input, target = input.to(device), target.to(device)

        output = model(input)
        loss = loss_fn(output, target)

        losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()

        loss.backward()

        if args.clip_grad is not None:
            dispatch_clip_grad(
                # model_parameters(model, exclude_head='agc' in args.clip_mode),
                model.parameters(),
                value=args.clip_grad,
                mode=args.clip_mode
            )
        optimizer.step()

        num_updates += 1
        batch_time_m.update(time.time() - end)

        if last_batch or (batch_idx+1) % args.log_interval == 0 or batch_idx == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            _logger.info(
                'Train: Epoch{:4d}/{:4d} [{:>4d}/ {}]  '
                'Loss: {loss.avg:3e}  '
                'Time: {batch_time.val:.2f}s, {rate:.2f}/s  '
                '({batch_time.avg:.2f}s, {rate_avg:.2f}/s)  '
                'LR: {lr:.3e}  '
                .format(
                    epoch+1, args.epochs,
                    batch_idx+1, len(loader),
                    loss=losses_m,
                    batch_time=batch_time_m,
                    rate=input.size(0) / batch_time_m.val,
                    rate_avg=input.size(0) / batch_time_m.avg,
                    lr=lr,
                ))

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    return OrderedDict([('loss', losses_m.avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device,
):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = (batch_idx == len(loader) - 1)

            input = input.to(device)
            target = target.to(device)

            output = model(input)

            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            reduced_loss = loss.data

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if last_batch:
                log_name = 'Test'
                _logger.info(
                    '{0} : [{1}/{2}]  '
                    'Time: {batch_time.avg:.3f}  '
                    'Loss: {loss.avg:.3e}  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    .format(
                        log_name, batch_idx, len(loader)-1,
                        batch_time=batch_time_m,
                        loss=losses_m,
                        top1=top1_m,
                    )
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

    return metrics


if __name__ == '__main__':
    main()

