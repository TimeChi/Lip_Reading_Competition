import os
import time
import shutil
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_

from eco_dataset import TSNDataSet,TSNDataSet_infer
from ops.models import TSN
from ops.transforms import *
import opts
from opts import parser
from ops.utils import AverageMeter, accuracy
from ops.temporal_shift import make_temporal_pool
from torch.nn.init import constant_, xavier_uniform_, xavier_normal_

best_prec1 = 0


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        xavier_normal_(m.weight.data)
        constant_(m.bias.data, 0.0)
    elif classname.find('layer') != -1:
        xavier_normal_(m.weight.data)
        constant_(m.bias.data, 0.0)
    elif classname.find('bn') != -1:
        constant_(m.weight.data)
        constant_(m.bias.data, 0.0)


def main():
    global args, best_prec1
    args = parser.parse_args()
    num_class = opts.num_class

    full_arch_name = args.arch
    if args.shift:
        full_arch_name += '_shift{}_{}'.format(args.shift_div, args.shift_place)
    if args.temporal_pool:
        full_arch_name += '_tpool'
    args.store_name = '_'.join(
        ['TSM', args.dataset, args.modality, full_arch_name, args.consensus_type, 'segment%d' % args.num_segments,
         'e{}'.format(args.epochs)])
    if args.pretrain != 'imagenet':
        args.store_name += '_{}'.format(args.pretrain)
    if args.lr_type != 'step':
        args.store_name += '_{}'.format(args.lr_type)
    if args.non_local > 0:
        args.store_name += '_nl'
    print('storing name: ' + args.store_name)

    # check_rootfolders()

    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type,
                dropout=args.dropout,
                img_size=args.img_size,
                partial_bn=not args.no_partialbn,
                pretrain=args.pretrain,
                is_shift=args.shift, shift_div=args.shift_div, shift_place=args.shift_place,
                fc_lr5=True,
                temporal_pool=args.temporal_pool,
                non_local=args.non_local)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation(
        flip=False if 'something' in args.dataset or 'jester' in args.dataset else True)

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    model.apply(weights_init)

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        if args.temporal_pool:  # early temporal pool so that we can load the state_dict
            make_temporal_pool(model.module.base_model, args.num_segments)
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(args.resume, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    if args.temporal_pool and not args.resume:
        make_temporal_pool(model.module.base_model, args.num_segments)

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    fr_r = open(opts.NUM_LABEL_R, 'r+')
    w2n = eval(fr_r.read())
    fr_r.close()

    fr = open(opts.NUM_LABEL, 'r+')
    n2w = eval(fr.read())
    fr.close()

    train_loader, val_loader, test_loader = None, None, None
    if args.mode != 'test':
        lip_dict, video_list = opts.file_deal(opts.TRAIN_DATA, w2n)
        train_num = int(len(video_list) * 0.95)
        train_loader = torch.utils.data.DataLoader(
            TSNDataSet(opts.TRAIN_DATA, args.mode, num_segments=args.num_segments,
                       img_size=args.img_size,
                       lip_dict=lip_dict,
                       video_list=video_list[:train_num],
                       transform=torchvision.transforms.Compose([
                           train_augmentation,
                           GroupMultiScaleCrop(args.img_size, [1, .875, .75, .66]),
                           Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                           normalize,
                       ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            TSNDataSet(opts.TRAIN_DATA, args.mode, num_segments=args.num_segments,
                       img_size=args.img_size,
                       lip_dict=lip_dict,
                       video_list=video_list[train_num:],
                       transform=torchvision.transforms.Compose([
                           GroupScale(int(scale_size)),
                           GroupCenterCrop(crop_size),
                           Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                           normalize,
                       ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    else:
        lip_dict, video_list = opts.file_deal(opts.TEST_DATA, w2n)
        test_loader = torch.utils.data.DataLoader(
            TSNDataSet_infer(opts.TEST_DATA, num_segments=args.num_segments,
                       img_size=args.img_size,
                       lip_dict=lip_dict,
                       video_list=video_list,
                       transform=torchvision.transforms.Compose([
                           GroupScale(int(scale_size)),
                           GroupCenterCrop(crop_size),
                           Stack(roll=(args.arch in ['BNInception', 'InceptionV3'])),
                           ToTorchFormatTensor(div=(args.arch not in ['BNInception', 'InceptionV3'])),
                           normalize,
                       ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    if args.mode == 'test':
        if args.sub == 'sub':
            inference(test_loader, model, n2w)
        else:
            inferencefusion(test_loader, model, n2w)
        return


    # 开始训练
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_type, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, epoch, n2w)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            output_best = 'Best Prec@1: %.3f\n' % (best_prec1)
            print(output_best)

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    torch.set_grad_enabled(True)

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, name) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        # input_var = torch.autograd.Variable(input)
        # target_var = torch.autograd.Variable(target)
        input_var = input
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5,
                lr=optimizer.param_groups[-1]['lr'] * 0.1))  # TODO
            print(output)


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    torch.set_grad_enabled(False)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target, name) in enumerate(val_loader):
            # discard final batch
            if i == len(val_loader) - 1:
                break
            target = target.cuda(async=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if epoch > int(args.epochs * 0.7):
                batch_out = torch.argmax(output, dim=1)
                batch_reslut = (batch_out != target).tolist()
                for i in range(len(batch_reslut)):
                    if batch_reslut[i] == 1:
                        print(name[i])

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)

    output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
              .format(top1=top1, top5=top5, loss=losses))
    print(output)

    return top1.avg


def inference(test_loader, model, num2words):
    # In PyTorch 0.4, "volatile=True" is deprecated.
    torch.set_grad_enabled(False)
    # switch to evaluate mode
    model.eval()
    flag = False
    with torch.no_grad():
        for i, (input, record) in enumerate(test_loader):
            # discard final batch
            if i == len(test_loader):
                break
            input_var = input
            # compute output
            output = model(input_var)
            batch_out = torch.argmax(output, dim=1)
            # final
            if flag == False:
                final = batch_out
                final_record = record
                flag = True
            else:
                final = torch.cat((final, batch_out), 0)
                final_record.extend(record)  # 得到文件名
            # print(final)
            # print('record:',final_record)
    final_list = final.tolist()

    words_list = [num2words[x] for x in final_list]  # 得到文字标签
    import pandas as pd
    result = pd.DataFrame()
    result['file'] = final_record
    result['label'] = words_list
    result.to_csv(opts.TEST_SUB, index=None, header=None)


def inferencefusion(test_loader, model, num2words):
    torch.set_grad_enabled(False)
    # switch to evaluate mode
    model.eval()
    flag = False
    with torch.no_grad():
        for i, (input, record) in enumerate(test_loader):
            # discard final batch
            if i == len(test_loader):
                break
            input_var = input
            # compute output
            output = model(input_var)
            batch_out = torch.argmax(output, dim=1)
            output = torch.nn.functional.softmax(output, dim=1)
            # final
            if flag == False:

                file_prob = output
                final = batch_out
                final_record = record
                flag = True
            else:
                file_prob = torch.cat((file_prob, output), 0)
                final = torch.cat((final, batch_out), 0)
                final_record.extend(record)  # 得到文件名
            # print(final)
            # print('record:',final_record)
    prob_list = file_prob.tolist()
    prob_final_list = [max(item) for item in prob_list]
    final_list = final.tolist()

    words_list = [num2words[x] for x in final_list]  # 得到文字标签
    import pandas as pd
    result = pd.DataFrame()
    result['file'] = final_record
    result['label'] = words_list
    result['prob'] = prob_final_list
    result.to_csv(opts.TEST_SUB, index=None, header=None)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, "epoch", str(state['epoch']), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)


def adjust_learning_rate_eco(optimizer, epoch, lr_steps, exp_num):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    decay = 0.1 ** (exp_num)
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = args.lr * decay
        decay = args.weight_decay
    elif lr_type == 'cos':
        import math
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * epoch / args.epochs))
        decay = args.weight_decay
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def check_rootfolders():
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)


if __name__ == '__main__':
    main()
