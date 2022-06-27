import sys
sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')


import argparse
import time
from nas.classification_darts.config import C
from nas.classification_darts.classify_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--method', type=str, required=True, help='location of the data corpus')
    parser.add_argument('--dataset', type=str, required=True, help='location of the data corpus')
    args = parser.parse_args()

    log_info = read_offline_log_to_string('{}/{}/logdir/Offline_{}_{}.log'.format(
        C.cache_dir, args.method, args.method, args.dataset))
    log_pointer = 0
    write_online_log_to_file(
        '{}/{}/logdir/Online_{}_{}.log'.format(
            C.cache_dir, args.method, args.method, args.dataset), log_info, log_pointer,
        write_length=50, recreate=True
    )

    while True:
        log_pointer += 1
        write_online_log_to_file(
            '{}/{}/logdir/Online_{}_{}.log'.format(
                C.cache_dir, args.method, args.method, args.dataset), log_info, log_pointer,
            write_length=50, recreate=False
        )
        time.sleep(1)
