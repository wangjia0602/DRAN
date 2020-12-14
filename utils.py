import sys, time
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Tracker(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.val = value
        self.sum += value
        self.count += 1
        self.avg = self.sum / self.count


def progress_bar(batch_idx, num_batches, loss):
    # progress bar
    prg = int(batch_idx / num_batches * 100)
    blc = int(prg / 5)
    sys.stdout.write('\r' + '{:>3.0f}'.format(prg) + '% |')
    sys.stdout.write('=' * (blc) + '>' + ' ' * (20 - blc))
    sys.stdout.write('|' + ' Train Loss : {:.4f}'.format(loss))
    sys.stdout.flush()
    if (batch_idx + 1) == num_batches:
        sys.stdout.write('\r' + '100% |' + '=' * 20 + '>')
        sys.stdout.write('|' + ' Train Loss : {:.4f}'.format(loss) + '\n')
        sys.stdout.flush()


def psnr(input, target):
    # peak signal-to-noise ratio
    return 10 * torch.log10(255 * 255 / F.mse_loss(input, target))


def plot(save_dir, stats, epoch):
    x = [i + 1 for i in range(epoch)]

    # train_loss & valid_loss
    plt.plot(x, stats['train_loss'], label='train_loss')
    plt.plot(x, stats['content_loss'], label='content_loss')
    plt.plot(x, stats['perceptual_loss'], label='perceptual_loss')

    plt.plot(x, stats['loss'], label='loss')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('{}/loss.png'.format(save_dir))
    plt.clf()

    # psnr
    plt.plot(x, stats['valid_psnr'])
    plt.xlabel('Epoch')
    plt.ylabel('PSNR(dB)')
    plt.savefig('{}/psnr.png'.format(save_dir))
    plt.clf()

    # 如何读取json
    # with open('stats.json','r') as f:
    #     stats = json.load(f)
    # 将stats保存为.txt文件
    # train_loss = stats['train_loss']
    # with open ('./../train_loss.txt','w') as f:
    #     for i in range(len(train_loss)):
    #         f.write(str(train_loss[i])+'\n')


def save_args(args, save_dir):
    args_dict = args.__dict__
    with open('{}/args.txt'.format(save_dir), 'w') as f:
        for key, value in args_dict.items():
            f.write(f'{key} = {value}\n')


if __name__ == '__main__':
    for i in range(30):
        progress_bar(i, 30, 128.14534)
        time.sleep(0.5)


# 未启用功能块
def optimizer(args, my_model):
    # 这个操作是否有必要？是否能减少计算量？
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    optimizer_function = None
    kwargs = {}
    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}

    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {'betas': (args.beta1, args.beta2),
                  'eps': args.epsilon}

    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['lr'] = arg.lr
    kwargs['weight_decay'] = args.weight_decay

    return optimizer_function(trainable, **kwargs)
