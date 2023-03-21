from training.GPT_ONLY import GPT_ONLY
import argparse
import yaml
from easydict import EasyDict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of Music2Dance')
    parser.add_argument('--config', default='configs/gpt_beat_align.yaml')
    # exclusive arguments
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')
    group.add_argument('--visgt', action='store_true')
    group.add_argument('--anl', action='store_true')
    group.add_argument('--sample', action='store_true')

    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in vars(args).items():
        config[k] = v

    config = EasyDict(config)
    agent = GPT_ONLY(config)

    if args.train:
        agent.train()
    elif args.eval:
        agent.eval()

    else:
        agent.train()


if __name__ == '__main__':
    main()
