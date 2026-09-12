import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='CEPM J-HMDB-21 training options')

    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--epochs', type=int, default=1, help='number of total epochs to run')
    parser.add_argument('--model_name', type=str, default='i3d', help='model name')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--sup_loc_loss', type=str, default='dice', help='dice or iou loss')
    parser.add_argument('--exp_id', type=str, default='debug', help='experiment name')

    parser.add_argument('--txt_file_label', type=str, default='jhmdb_classes_list_per_20_labeled.txt', help='labeled subset file')
    parser.add_argument('--txt_file_unlabel', type=str, default='jhmdb_classes_list_per_80_unlabeled.txt', help='unlabeled subset file')
    parser.add_argument('--jhmdb_dataset_path', type=str, default='path/to/JHMDB/data/Videos/ReCompress_Videos', help='J-HMDB dataset path')
    parser.add_argument('--jhmdb_exp_save_path', type=str, default=None, help='experiment output path')

    parser.add_argument('--const_loss', type=str, default='l2', help='consistency loss type')
    parser.add_argument('--wt_loc', type=float, default=1, help='supervised localization loss weight')
    parser.add_argument('--wt_cls', type=float, default=1, help='supervised classification loss weight')
    parser.add_argument('--wt_cons', type=float, default=0.1, help='consistency loss weight')

    parser.add_argument('-at', '--aug_type', type=int, help='0-spatial, 1-temporal, 2-both')
    parser.add_argument('-ema', '--ema_val', type=float, help='EMA decay')

    parser.add_argument('--thresh_epoch', type=int, default=11, help='epoch to introduce pseudo labels')
    parser.add_argument('--ramp_thresh', type=int, default=0, help='ramp up consistency loss until this epoch')
    parser.add_argument('--recon_start_epoch', type=int, default=11, help='epoch to start reconstruction loss')
    parser.add_argument('--beta', type=float, default=2.0, help='reconstruction loss weight')
    parser.add_argument('--entropy_thresh_min', type=float, default=None, help='minimum entropy threshold')
    parser.add_argument('--entropy_thresh_max', type=float, default=None, help='maximum entropy threshold')

    parser.add_argument('--queue_size', type=int, default=128, help='size of the memory bank')
    parser.add_argument('--num_attention_layers', type=int, default=2, help='number of reconstruction attention layers')
    parser.add_argument(
        '--fusion_mode',
        type=str,
        default='diff_cross_attention',
        choices=['linear', 'cross_attention', 'diff_cross_attention'],
        help='feature fusion mode used in the reconstruction module',
    )

    parser.add_argument('-burn', '--burn_in', action='store_true', help='use burn in weights')
    parser.add_argument('-bw', '--burn_wts', type=str, default='debug', help='burn-in experiment name')

    parser.add_argument('--dice', action='store_true', help='L2+dice')
    parser.add_argument('--opt1', action='store_true', help='optional flag 1')
    parser.add_argument('--opt2', action='store_true', help='optional flag 2')
    parser.add_argument('--opt3', action='store_true', help='optional flag 3')
    parser.add_argument('--opt4', action='store_true', help='optional flag 4')
    parser.add_argument('--opt5', action='store_true', help='optional flag 5')
    parser.add_argument('--scheduler', action='store_true', help='use lr scheduler')

    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training')
    parser.add_argument('--seed_data', type=int, default=47, help='seed for data split usage')

    args = parser.parse_args()
    return args
