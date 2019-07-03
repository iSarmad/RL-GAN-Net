
import os
import time
import torch
import datetime

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np

from sagan_models import Generator, Discriminator
from utils import *
from collections import OrderedDict


class Trainer(object):
    def __init__(self, data_loader, args,train_loader,model_decoder,chamfer,vis_Valida):

        # decoder settings
        self.model_decoder = model_decoder
        self.chamfer =chamfer
        self.vis = vis_Valida
        self.j =0

        # Data loader
      #  self.data_loader = data_loader
        self.train_loader = train_loader # TODO

        # exact model and loss
        self.args = args
        self.model = args.model
        self.adv_loss = args.adv_loss

        # Model hyper-parameters
        self.imsize = args.imsize
        self.g_num = args.g_num
        self.z_dim = args.z_dim
        self.g_conv_dim = args.g_conv_dim
        self.d_conv_dim = args.d_conv_dim
        self.parallel = args.parallel

        self.lambda_gp = args.lambda_gp
        self.total_step = args.total_step
        self.d_iters = args.d_iters
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.lr_decay = args.lr_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.pretrained_model = args.pretrained_model

        self.dataset = args.dataset
        self.use_tensorboard = args.use_tensorboard
        self.image_path = args.image_path
        self.log_path = args.log_path
        self.model_save_path = args.model_save_path
        self.sample_path = args.sample_path
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step
        self.version = args.version

        # Path
        self.log_path = os.path.join(args.log_path, self.version)
        self.sample_path = os.path.join(args.sample_path, self.version)
        self.model_save_path = os.path.join(args.model_save_path, self.version)

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()



    def train(self):

        # Data iterator
      #  data_iter = iter(self.data_loader)
        train_iter = iter(self.train_loader) # TODO

       # step_per_epoch = len(self.data_loader)
        train_step_per_epoch = len(self.train_loader) # TODO


       # model_save_step = int(self.model_save_step * step_per_epoch)
        model_save_step = int(self.model_save_step * train_step_per_epoch) # TODO

        # Fixed input for debugging
        fixed_z_np = np.arange(-self.args.max_action,self.args.max_action,(self.args.max_action*2)/50)#self.batchsize replace with 10
        fixed_z_n = tensor2var(torch.FloatTensor(fixed_z_np,))
        fixed_z = fixed_z_n.unsqueeze(1)
       # fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))
    #    fixed_z = tensor2var(torch.)
        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        # Start time
        start_time = time.time()
        for step in range(start, self.total_step):

            # ================== Train D ================== #
            self.D.train()
            self.G.train()

            try:
              #  real_images, _ = next(data_iter)
                real_images = next(train_iter)  # TODO

            except:
              #  data_iter = iter(self.data_loader)
                train_iter = iter(self.train_loader) # TODO

               # real_images, _ = next(data_iter)
                real_images = next(train_iter)

            # Compute loss with real images
            # dr1, dr2, df1, df2, gf1, gf2 are attention scores
            real_images = tensor2var(real_images)
            d_out_real,dr1 = self.D(real_images)#,dr2
            if self.adv_loss == 'wgan-gp':
                d_loss_real = - torch.mean(d_out_real)
            elif self.adv_loss == 'hinge':
                d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

            # apply Gumbel Softmax
            z = tensor2var((torch.randn(real_images.size(0), self.z_dim)))
            fake_images,gf1 = self.G(z) #,gf2
            d_out_fake,df1 = self.D(fake_images) #,df2

            if self.adv_loss == 'wgan-gp':
                d_loss_fake = d_out_fake.mean()
            elif self.adv_loss == 'hinge':
                d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()


            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()


            if self.adv_loss == 'wgan-gp':
                # Compute gradient penalty
                alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                out,_= self.D(interpolated) # TODO "_"

                grad = torch.autograd.grad(outputs=out,
                                           inputs=interpolated,
                                           grad_outputs=torch.ones(out.size()).cuda(),
                                           retain_graph=True,
                                           create_graph=True,
                                           only_inputs=True)[0]

                grad = grad.view(grad.size(0), -1)
                grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                # Backward + Optimize
                d_loss = self.lambda_gp * d_loss_gp

                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

            # ================== Train G and gumbel ================== #
            # Create random noise
            z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
            fake_images,_ = self.G(z) # _

            # Compute loss with fake images
            g_out_fake,_ = self.D(fake_images)  # batch x n  TODO "_"
            if self.adv_loss == 'wgan-gp':
                g_loss_fake = - g_out_fake.mean()
            elif self.adv_loss == 'hinge':
                g_loss_fake = - g_out_fake.mean()

            self.reset_grad()
            g_loss_fake.backward()
            self.g_optimizer.step()


            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, d_out_fake: {:.4f}, "
                      " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                      format(elapsed, step + 1, self.total_step, (step + 1),
                             self.total_step , d_loss_real.data[0], d_loss_fake.data[0],self.total_step , d_loss_real.data[0],self.total_step , d_loss_real.data[0]))
                          #   self.G.attn1.gamma.mean().data[0], self.G.attn2.gamma.mean().data[0] ))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images,_= self.G(fixed_z) #TODO "_"

                encoded = fake_images.contiguous().view(50,128) # 64,[1024,128] Gan output dims

                pc_1 = self.model_decoder(encoded) #   real_images.contiguous().view(64,128)
                #pc_1_temp = pc_1[0, :, :]

                epoch =0;
                for self.j in range(0,50):#self.bacth_size
                    pc_1_temp = pc_1[self.j, :, :]
                    test = fixed_z.detach().cpu().numpy()
                    test1 = np.asscalar(test[self.j,0])
                   # test1 = 0
                    visuals = OrderedDict(
                        [('Validation Predicted_pc', pc_1_temp.detach().cpu().numpy())])
                    self.vis[self.j].display_current_results(visuals, epoch, step,z =str(test1))


                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))

    def build_model(self):

        self.G = Generator(self.batch_size,self.imsize, self.z_dim, self.g_conv_dim).cuda()
        self.D = Discriminator(self.batch_size,self.imsize, self.d_conv_dim).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):
        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))
