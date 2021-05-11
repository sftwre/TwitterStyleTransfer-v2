"""
Modified code from https://github.com/fastnlp/style-transformer
"""
import os
import torch
import yaml
from argparse import ArgumentParser
from data import load_dataset
from transformer import StyleTransformer, Discriminator
from train import train, auto_eval

class Config():

    conf = '../config.yaml'

    with open(conf, 'r') as file:
        config = yaml.safe_load(file.read())

    data_path = os.path.join(config.get('DATA_PATH'), 'individual')
    log_dir = 'runs/exp'
    save_path = os.path.join(config.get('PROJECT_DIR'), 'save')
    pretrained_embed_path = os.path.join(config.get('PROJECT_DIR'), 'embedding/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator_method = 'Multi'  # 'Multi' or 'Cond'
    load_pretrained_embed = True
    min_freq = 3
    max_length = 58
    embed_size = 100
    d_model = 100
    h = 4
    num_styles = 4
    num_classes = num_styles + 1 if discriminator_method == 'Multi' else 2
    num_layers = 4
    batch_size = 64
    lr_F = 0.0001
    lr_D = 0.0001
    L2 = 0
    iter_D = 10
    iter_F = 5
    F_pretrain_iter = 500
    log_steps = 5
    eval_steps = 25
    learned_pos_embed = True
    dropout = 0
    drop_rate_config = [(1, 0)]
    temperature_config = [(1, 0)]

    slf_factor = 0.25
    cyc_factor = 0.5
    adv_factor = 1

    inp_shuffle_len = 0
    inp_unk_drop_fac = 0
    inp_rand_drop_fac = 0
    inp_drop_prob = 0


def main():

    parser = ArgumentParser()
    parser.add_argument('--train', dest='train_model', action='store_true', help='Flag to train model')
    parser.add_argument('--eval', dest='train_model', action='store_false', help='Flag to evaluate pre-trained model')
    parser.set_defaults(train_model=True)

    parser.add_argument('--step', required=False, type=int, help='Last global step of pre-trained model to load.')
    parser.add_argument('--save_path', required=False, type=str, help='Path to save directory.')

    args = parser.parse_args()
    train_model = args.train_model
    step = args.step
    save_path = args.save_path

    config = Config()
    train_iters, test_iters, vocab = load_dataset(config)
    print('Vocab size:', len(vocab))
    model_F = StyleTransformer(config, vocab).to(config.device)
    model_D = Discriminator(config, vocab).to(config.device)

    # load pre-trained models from save_path
    if (save_path is not None) and (step is not None):
        f_pth = os.path.join(config.config.get('PROJECT_DIR'), save_path, 'ckpts', f'{step}_F.pth')
        d_pth = os.path.join(config.config.get('PROJECT_DIR'), save_path, 'ckpts', f'{step}_D.pth')

        model_F.load_state_dict(torch.load(f_pth))
        model_D.load_state_dict(torch.load(d_pth))

    print(config.discriminator_method)

    if train_model:
        train(config, vocab, model_F, model_D, train_iters)
    else:
        auto_eval(config, vocab, model_F, test_iters, step, 1)


if __name__ == '__main__':
    main()
