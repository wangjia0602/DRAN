from argparse import ArgumentParser
from dataset_plus import load_test_dataset
from srnn import SRNN


def parse_args():
    parser = ArgumentParser(description='Test modle.')

    # Data parser
    parser.add_argument('--test_data_dir', type=str, default='./../dataset/benchmark/Set5',
                        help='test data dir')
    parser.add_argument('--result_dir', type=str, default='./../dataset/microscope_dataset/result',
                        help='result dir')
    parser.add_argument('--ckpt_file', type=str, default='./../ckpts/DeepDenseAttentionNetwork2020-10-06-05-13-03/epoch665-28.3516.pt',#
                        help='load model checkpoint file')
    parser.add_argument('--load_ckpt', type=bool, default=True,
                        help='load model checkpoint file')
    # Model parser (16,64)
    parser.add_argument('--para', type=int, default=3,
                        help='parameter')
    parser.add_argument('--n_resblocks', type=int, default=12,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--scale', type=int, default=4,
                        help='super resolution scale')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')
    parser.add_argument('--n_resgroups', type=int, default=24,
                        help='number of residual groups')
    parser.add_argument('--reduction', type=int, default=16,
                        help='number of feature maps reduction')

    # Training hyperparameters (16, 200)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=800,
                        help='number of epochs')
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda')
    parser.add_argument('--train', type=bool, default=False,
                        help='is')
    parser.add_argument('--transform', type=bool, default=False,
                        help='is')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local_rank')


    #####san
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--act', type=str, default='relu',
                        help='activation function')
    parser.add_argument('--shift_mean', default=True,
                        help='subtract pixel mean from the input')
    ####################################
    ########loss
    parser.add_argument('--content_loss_factor', type=float, default=1e-2,
                        help='content loss factor when training generator oriented')
    parser.add_argument('--perceptual_loss_factor', type=float, default=1,
                        help='perceptual loss factor when training '
                             'generator oriented')

    arg = parser.parse_args()

    return arg


if __name__ == '__main__':
    arg = parse_args()

    srnn = SRNN(arg, trainable=False)
    test_loader = load_test_dataset(arg)
    srnn.test(test_loader)
