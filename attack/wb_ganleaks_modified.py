import numpy as np
import os
import sys
import pickle
import argparse
from tqdm import tqdm
import torch
import models

### import tools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools/lpips_pytorch'))
from utils import *
import lpips as ps
from LBFGS_pytorch import FullBatchLBFGS
import torchvision.datasets as tvds
import torchvision.transforms as transforms
import torchvision

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../gan_models/vaegan'))
# from train import *

### Hyperparameters
LAMBDA2 = 0.2
LAMBDA3 = 0.001
LBFGS_LR = 0.015
RANDOM_SEED = 1000


#############################################################################################################
# get and save the arguments
#############################################################################################################
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-name', type=str, required=True,
                        help='the name of the current experiment (used to set up the save_dir)')
    parser.add_argument('--gan_model_dir', '-gdir', type=str, required=True,
                        help='directory for the Victim GAN model')
    # parser.add_argument('--pos_data_dir', '-posdir', type=str,
    #                     help='the directory for the positive (training) query images set')
    # parser.add_argument('--neg_data_dir', '-negdir', type=str,
    #                     help='the directory for the negative (testing) query images set')
    parser.add_argument('--data_num', '-dnum', type=int, default=100,
                        help='the number of query images to be considered')
    parser.add_argument('--train_num', '-tnum', type=int, default=100,
                        help='the number of total training set')
    parser.add_argument('--batch_size', '-bs', type=int, default=50,
                        help='batch size (should not be too large for better optimization performance)')
    parser.add_argument('--resolution', '-resolution', type=int, default=32,
                        help='generated image resolution')
    parser.add_argument('--initialize_type', '-init', type=str, default='random',
                        choices=['zero',  # 'zero': initialize the z to be zeros
                                 'random',  # 'random': use normal distributed initialization
                                 'nn',  # 'nn': use nearest-neighbor initialization
                                 ],
                        help='the initialization techniques')
    parser.add_argument('--nn_dir', '-ndir', type=str,
                        help='the directory for storing the fbb(KNN) results')
    parser.add_argument('--distance', '-dist', type=str, default='l2-lpips', choices=['l2', 'l2-lpips'],
                        help='the objective function type')
    parser.add_argument('--if_norm_reg', '-reg', action='store_true', default=True,
                        help='enable the norm regularizer')
    parser.add_argument('--maxfunc', '-mf', type=int, default=1000,
                        help='the maximum number of function calls (for scipy optimizer)')
    return parser.parse_args()


def check_args(args):
    '''
    check and store the arguments as well as set up the save_dir
    :param args: arguments
    :return:
    '''
    ## load dir
    assert os.path.exists(args.gan_model_dir)

    ## set up save_dir
    save_dir = os.path.join(os.path.dirname(__file__), 'gan_leaks_results/wb', args.exp_name)
    check_folder(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))
    pickle.dump(vars(args), open(os.path.join(save_dir, 'params.pkl'), 'wb'), protocol=2)

    return args, save_dir, args.gan_model_dir


#############################################################################################################
# main optimization function
#############################################################################################################
class LatentZ(torch.nn.Module):
    def __init__(self, init_val):
        super(LatentZ, self).__init__()
        self.z = torch.nn.Parameter(init_val.data)

    def forward(self):
        return self.z

    def reinit(self, init_val):
        self.z = torch.nn.Parameter(init_val.data)


class Loss(torch.nn.Module):
    def __init__(self, netG, distance, if_norm_reg=False, z_dim=256):
        super(Loss, self).__init__()
        self.distance = distance
        self.lpips_model = ps.LPIPS(net='vgg').cuda()
        self.netG = netG
        self.if_norm_reg = if_norm_reg
        self.z_dim = z_dim

        ### loss
        if distance == 'l2':
            print('Use distance: l2')
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])
            self.loss_lpips_fn = lambda x, y: 0.

        elif distance == 'l2-lpips':
            print('Use distance: lpips + l2')
            self.loss_lpips_fn = lambda x, y: self.lpips_model(x,y).view(-1)
            self.loss_l2_fn = lambda x, y: torch.mean((y - x) ** 2, dim=[1, 2, 3])

    def forward(self, z, x_gt, cond):
        # import pdb;pdb.set_trace()
        self.x_hat = self.netG(z, cond)
        self.loss_lpips = self.loss_lpips_fn(self.x_hat, x_gt)
        self.loss_l2 = self.loss_l2_fn(self.x_hat, x_gt)
        self.vec_loss = LAMBDA2 * self.loss_lpips + self.loss_l2

        if self.if_norm_reg:
            z_ = z.view(-1, self.z_dim)
            norm = torch.sum(z_ ** 2, dim=1)
            norm_penalty = (norm - self.z_dim) ** 2
            self.vec_loss += LAMBDA3 * norm_penalty

        return self.vec_loss


def optimize_z_lbfgs(loss_model,
                     init_val,
                     query_loader,
                     save_dir,
                     max_func,
                     mix = False,
                     loader2 = None):
    ### store results
    all_loss = []
    all_z = []
    all_x_hat = []

    loader = query_loader if not mix else zip(query_loader,loader2)
    ### run the optimization for all query data
    for i, data in tqdm(enumerate(loader)):
        save_dir_batch = os.path.join(save_dir, str(i))

        try:
            if not mix:
                x_gt = data[0].cuda()
                y = data[1][:,20].cuda()
                y_oh = torch.nn.functional.one_hot(y, Y_DIM)
            else:
                lam = np.random.beta(1,1)
                x_gt = (data[0][0] * lam + data[1][0] * (1-lam)).cuda()
                y1 = data[0][1][:,20].cuda()
                y_oh1 = torch.nn.functional.one_hot(y1, Y_DIM)
                y2 = data[1][1][:,20].cuda()
                y_oh2 = torch.nn.functional.one_hot(y2, Y_DIM)
                y_oh = y_oh1 * lam  +  y_oh2 * (1-lam)

            if os.path.exists(save_dir_batch):
                pass
            else:
                torchvision.utils.save_image(x_gt, os.path.join(check_folder(save_dir_batch), 'orig.png'))

                ### initialize z
                z = torch.randn((BATCH_SIZE, Z_DIM), requires_grad=True)
                # z = Variable(torch.FloatTensor(init_val[i * BATCH_SIZE:(i + 1) * BATCH_SIZE])).cuda()
                z_model = LatentZ(z).cuda()

                ### LBFGS optimizer
                optimizer = FullBatchLBFGS(z_model.parameters(), lr=LBFGS_LR, history_size=20, line_search='Wolfe',
                                           debug=False)

                ### optimize
                loss_progress = []

                def closure():
                    optimizer.zero_grad()
                    vec_loss = loss_model.forward(z_model.forward(), x_gt, y_oh)
                    vec_loss_np = vec_loss.detach().cpu().numpy()
                    loss_progress.append(vec_loss_np)
                    final_loss = torch.mean(vec_loss)
                    return final_loss

                for step in range(max_func):
                    loss_model.forward(z_model.forward(), x_gt, y_oh)
                    final_loss = closure()
                    final_loss.backward()

                    options = {'closure': closure, 'current_loss': final_loss, 'max_ls': 20}
                    obj, grad, lr, _, _, _, _, _ = optimizer.step(options)

                    if step == 0:
                        ### store init
                        x_hat_curr = loss_model.x_hat.data.cpu().numpy()
                        x_hat_curr = np.transpose(x_hat_curr, [0, 2, 3, 1])
                        vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()
                        visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize init
                        torchvision.utils.save_image(loss_model.x_hat, os.path.join(save_dir_batch, 'output0.png'))
                        

                    if step == max_func - 1:
                        vec_loss_curr = loss_model.vec_loss.data.cpu().numpy()
                        z_curr = z_model.z.data.cpu().numpy()
                        x_hat_curr = loss_model.x_hat.data.cpu().numpy()
                        x_hat_curr = np.transpose(x_hat_curr, [0, 2, 3, 1])

                        loss_lpips = loss_model.loss_lpips.data.cpu().numpy()
                        loss_l2 = loss_model.loss_l2.data.cpu().numpy()
                        save_files(save_dir_batch, ['l2', 'lpips'], [loss_l2, loss_lpips])

                        ### store results
                        torchvision.utils.save_image(loss_model.x_hat, os.path.join(save_dir_batch, 'final_output.png'))
                        visualize_progress(x_hat_curr, vec_loss_curr, save_dir_batch, step)  # visualize finale
                        all_loss.append(vec_loss_curr)
                        all_z.append(z_curr)
                        all_x_hat.append(x_hat_curr)

                        save_files(save_dir_batch,
                                   ['full_loss', 'z', 'xhat', 'loss_progress'],
                                   [vec_loss_curr, z_curr, x_hat_curr, np.array(loss_progress)])

        except KeyboardInterrupt:
            print('Stop optimization\n')
            break

    try:
        all_loss = np.concatenate(all_loss)
        all_z = np.concatenate(all_z)
        all_x_hat = np.concatenate(all_x_hat)
    except:
        all_loss = np.array(all_loss)
        all_z = np.array(all_z)
        all_x_hat = np.array(all_x_hat)
    return all_loss, all_z, all_x_hat


#############################################################################################################
# main
#############################################################################################################    
Z_DIM = 256 
Y_DIM = 2 
def main():
    args, save_dir, load_dir = check_args(parse_arguments())

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    ### set up Generator
    network_path = load_dir # os.path.join(load_dir, 'G_100000.pkl')
    netG = models.GanGenerator(z_dim=Z_DIM, y_dim=Y_DIM).cuda()
    netG.load_state_dict(torch.load(network_path))
    netG.eval()
    # Z_DIM = netG.deconv1.module.in_channels
    resolution = args.resolution

    ### define loss
    loss_model = Loss(netG, args.distance, if_norm_reg=False, z_dim=Z_DIM)

    ### initialization
    if args.initialize_type == 'zero':
        init_val = np.zeros((args.data_num, Z_DIM, 1, 1))
        init_val_pos = init_val
        init_val_neg = init_val

    elif args.initialize_type == 'random':
        np.random.seed(RANDOM_SEED)
        init_val_np = np.random.normal(size=(Z_DIM, 1, 1))
        init_val_np = init_val_np / np.sqrt(np.mean(np.square(init_val_np)) + 1e-8)
        init_val = np.tile(init_val_np, (args.data_num, 1, 1, 1)).astype(np.float32)
        init_val_pos = init_val
        init_val_neg = init_val

    elif args.initialize_type == 'nn':
        idx = 0
        init_val_pos = np.load(os.path.join(args.nn_dir, 'pos_z.npy'))[:, idx, :]
        init_val_pos = np.reshape(init_val_pos, [len(init_val_pos), Z_DIM, 1, 1])
        init_val_neg = np.load(os.path.join(args.nn_dir, 'neg_z.npy'))[:, idx, :]
        init_val_neg = np.reshape(init_val_neg, [len(init_val_neg), Z_DIM, 1, 1])
    else:
        raise NotImplementedError
    trans_n = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    trans_crop = transforms.CenterCrop(128)
    trnas_resize = transforms.Resize(64)
    trans_tensor = transforms.ToTensor()
    trans = transforms.Compose([trans_crop, trnas_resize, trans_tensor])
    # trans = transforms.Compose([trnas_resize, trans_tensor])
    dataset = torchvision.datasets.CelebA('../dataset/', split= 'train', target_type= 'attr', transform = trans, target_transform = None, download = False)
    torch.manual_seed(0)
    pos_data, rest_data = torch.utils.data.random_split(dataset, [args.train_num, len(dataset)-args.train_num])
    pos_data, _ = torch.utils.data.random_split(pos_data, [args.data_num, len(pos_data)-args.data_num])
    neg_data, _ = torch.utils.data.random_split(rest_data, [args.data_num, len(rest_data)-args.data_num])

    pos_loader = torch.utils.data.DataLoader(pos_data, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
    pos_loader2 = torch.utils.data.DataLoader(pos_data, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)

    ### positive ###
    query_loss, query_z, query_xhat = optimize_z_lbfgs(loss_model,
                                                       init_val_pos,
                                                       pos_loader,
                                                       check_folder(os.path.join(save_dir, 'pos_results')),
                                                       args.maxfunc,
                                                       mix=True,
                                                       loader2=pos_loader2)
    save_files(save_dir, ['pos_loss'], [query_loss])

    ### negative ###

    neg_loader = torch.utils.data.DataLoader(neg_data, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
    query_loss, query_z, query_xhat = optimize_z_lbfgs(loss_model,
                                                       init_val_neg,
                                                       neg_loader,
                                                       check_folder(os.path.join(save_dir, 'neg_results')),
                                                       args.maxfunc)
    save_files(save_dir, ['neg_loss'], [query_loss])


if __name__ == '__main__':
    main()
