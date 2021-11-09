import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import os
from tensorboardX import SummaryWriter
import argparse
from datetime import datetime
from shutil import copyfile
import torchvision.transforms
from tqdm import tqdm
from util.SpadDataset import SpadDataset, RandomCrop, ToTensor
from models import DenoiseModel,SPADnet
from new_model_level4 import UnetPP3D
from q_new_model_level3 import Q_UnetPP3D 
import skimage.io
#%%
cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def tv(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
           
#--------------training function------------------
def train(model, device,train_loader, val_loader, optimizer, n_iter,
          lambda_tv, epoch, logfile, val_every=10, save_every=100,
          model_name='Q_UnetPP3D'):
    loss_list=[]
    for sample in tqdm(train_loader):#progress bar
        model.train()
        spad = sample['spad']
        rates = sample['rates']
        bins = sample['bins']

        spad_var = Variable(spad.type(dtype))
        depth_var = Variable(bins.type(dtype))
        rates_var = Variable(rates.type(dtype))
        
        spad_var, rates_var = spad_var.to(device), rates_var.to(device)
        
        # Run the model forward to compute scores and loss.
        denoise_out, sargmax = model(spad_var)
        lsmax_denoise_out = torch.nn.LogSoftmax(dim=1)(denoise_out).unsqueeze(1)
        # Compare denoised histogram(logsoftmaxed) with rate
        kl_loss = torch.nn.KLDivLoss()(lsmax_denoise_out, rates_var)
        #poisson_loss=torch.nn.PoissonNLLLoss()(lsmax_denoise_out, rates_var)
        tv_reg = lambda_tv * tv(sargmax)
        loss = kl_loss + tv_reg
        #loss = poisson_loss + tv_reg
        
        # Run the model backward and take a step using the optimizer.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_iter += 1
        
        #plot_dynmc(loss_list)
        print('\n',loss.item())
        
        # log in tensorboard
        writer.add_scalar('data/train_loss',kl_loss.data.cpu().numpy(), n_iter)
        writer.add_scalar('data/train_rmse', np.sqrt(np.mean((
                      sargmax.data.cpu().numpy() - depth_var.data.cpu().numpy())**2) /
                      sargmax.size()[0]), n_iter)
        
        if (n_iter + 1) % val_every == 0:
            model.eval()
            evaluate(model, val_loader, n_iter, model_name)

        if (n_iter + 1) % save_every == 0:
            save_checkpoint({
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter,
                 }, filename=logfile +
                 '/epoch_{}_{}.pth'.format(epoch, n_iter))

    return n_iter

#--------------Validate and plot function-------------
def evaluate(model, val_loader, n_iter, model_name):
    model.eval()
    sample = iter(val_loader).next()
    spad = sample['spad']
    bins = sample['bins']
    rates = sample['rates']

    spad_var = Variable(spad.type(dtype))
    depth_var = Variable(bins.type(dtype))
    rates_var = Variable(rates.type(dtype))

    denoise_out, sargmax = model(spad_var)

    lsmax_denoise_out = torch.nn.LogSoftmax(dim=1)(
        denoise_out).unsqueeze(1)
    kl_loss = torch.nn.KLDivLoss()(lsmax_denoise_out, rates_var)

    writer.add_scalar('data/val_loss',
                      kl_loss.data.cpu().numpy(), n_iter)
    writer.add_scalar('data/val_rmse', np.sqrt(np.mean((
                      sargmax.data.cpu().numpy() - depth_var.data.cpu().numpy())**2) /
                      sargmax.size()[0]), n_iter)

    im_est_depth = sargmax.data.cpu()[0:4, :, :, :].repeat(
                   1, 3, 1, 1)
    im_depth_truth = depth_var.data.cpu()[0:4, :, :].repeat(
                     1, 3, 1, 1)
    to_display = torch.cat((
                 im_est_depth, im_depth_truth), 0)
    im_out = torchvision.utils.make_grid(to_display,
                                         normalize=True,
                                         scale_each=True,
                                         nrow=4)
    writer.add_image('image', im_out, n_iter)
    return

#---------------main function------------------

train_files = 'util/train_intensity.txt'
val_files = 'util/val_intensity.txt'
noise_param_idx=2
batch_size = 4
workers = 0
epochs = 4
lr = 1e-4
lambda_tv = 1e-5

save_every = 100#save training ck point
print_every = 10#save val ck point
logdir = './logging'
resume = 'logging'#revise this if training interrupted
model_name = Q_UnetPP3D
log_name = 'Q_level_3W2A8'

def main():
    # set gpu
    en_cuda = torch.cuda.is_available()
    device = torch.device("cuda")
    # tensorboard log file
    global writer
    now = datetime.now()
    logfile = logdir + '/' + log_name + '_date_' + now.strftime('%m_%d-%H_%M') + '/'
    writer = SummaryWriter(logfile)
    print('=> Tensorboard logging to {}'.format(logfile))

    #model = eval(model_name + '()')
    model=Q_UnetPP3D()
    if en_cuda:
        model=model.cuda()
    model.type(dtype)

    # initialize optimization tools
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, 1e-4)

    # datasets and dataloader
    train_dataset = SpadDataset(train_files, noise_param_idx,32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=workers,
                              pin_memory=True)
    val_dataset = SpadDataset(val_files, noise_param_idx,32)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers,
                            pin_memory=True)
    
    if os.path.isfile(resume):#load ckpt file
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        #load start epoch
        try:
            start_epoch = checkpoint['epoch']
        except KeyError as err:
            start_epoch = 0
            print('=> Can''t load start epoch, setting to zero')
        #load learning rate
        try:
            lr = checkpoint['lr']
            print('=> Loaded learning rate {}'.format(lr))
        except KeyError as err:
            print('=> Can''t load learning rate, setting to default')
        #load parameters
        try:
            ckpt_dict = checkpoint['state_dict']
        except KeyError as err:
            ckpt_dict = checkpoint
            
        model_dict = model.state_dict()
        for k in ckpt_dict.keys():
            model_dict.update({k: ckpt_dict[k]})
        model.load_state_dict(model_dict)
        print('=> Loaded {}'.format(resume))
        #load optimizer
        try:
            optimizer_dict = optimizer.state_dict()
            ckpt_dict = checkpoint['optimizer']
            for k in ckpt_dict.keys():
                optimizer_dict.update({k: ckpt_dict[k]})
            optimizer.load_state_dict(optimizer_dict)
        except (ValueError, KeyError) as err:
            print('=> Unable to resume optimizer from checkpoint')

        # set optimizer learning rate
        for g in optimizer.param_groups:
            g['lr'] = lr
        try:
            n_iter = checkpoint['n_iter']
        except KeyError:
            n_iter = 0

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, start_epoch))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    # run training epochs
    print('=> starting training')
    n_iter = 0
    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        print('epoch: {}, lr: {}'.format(epoch, optimizer.param_groups[0]['lr']))
        
        n_iter = train(model, device,train_loader, val_loader, optimizer, n_iter,
                           lambda_tv, epoch, logfile,
                           val_every=print_every,
                           save_every=save_every,
                           model_name=model_name)
        
        #np.save('{}_epoch loss'.format(epoch),loss_list)
        # decrease the learning rate
        for g in optimizer.param_groups:
            g['lr'] *= 0.9

        save_checkpoint({
            'lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': n_iter,
             }, filename=logfile + '/epoch_{}_{}.pth'.format(epoch, n_iter))


if __name__ == '__main__':
    main()