# Written by Muhammad Sarmad
# Date : 23 August 2018



from RL_params import *



np.random.seed(5)
#torch.manual_seed(5)

dataset_names = sorted(name for name in Datasets.__all__)
model_names = sorted(name for name in models.__all__)

def evaluate_policy(policy,valid_loader,env,args, eval_episodes=6,render = False):
    avg_reward = 0.
    env.reset(epoch_size=len(valid_loader),figures=8) # reset the visdom and set number of figures

    #for i,(input) in enumerate(valid_loader):
    for i in range (0,eval_episodes):
        try:
            input = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(valid_loader)
            input = next(dataloader_iterator)

       # data_iter = iter(valid_loader)
       # input = data_iter.next()
        #action_rand = torch.randn(args.batch_size, args.z_dim)
        obs =env.agent_input(input)# env(input, action_rand)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = policy.select_action(np.array(obs))
            action= torch.tensor(action).cuda().unsqueeze(dim=0)
            new_state, _, reward, done, _ = env( input, action,render=render,disp =True)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break;

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward

def test_policy(policy,valid_loader,env,args, eval_episodes=12,render = True):
    avg_reward = 0.
    env.reset(epoch_size=len(valid_loader),figures=12) # reset the visdom and set number of figures

    #for i,(input) in enumerate(valid_loader):
    for i in range (0,eval_episodes):
        try:
            input = next(dataloader_iterator)
        except:
            dataloader_iterator = iter(valid_loader)
            input = next(dataloader_iterator)

       # data_iter = iter(valid_loader)
       # input = data_iter.next()
        #action_rand = torch.randn(args.batch_size, args.z_dim)
        obs =env.agent_input(input)# env(input, action_rand)
        done = False

        while not done:
          # Action By Agent and collect reward
            action = policy.select_action(np.array(obs))
            action= torch.tensor(action).cuda().unsqueeze(dim=0)
            new_state, _, reward, done, _ = env( input, action,render=render,disp =True)
            avg_reward += reward

        if i+1 >= eval_episodes:
            break;

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")

    return avg_reward






def main(args,vis_Valid,vis_Valida):
    """ Transforms/ Data Augmentation Tec """
    co_transforms = pc_transforms.Compose([
        #  pc_transforms.Delete(num_points=1466)
          pc_transforms.Jitter_PC(sigma=0.01,clip=0.05),
         # pc_transforms.Scale(low=0.9,high=1.1),
        #   pc_transforms.Shift(low=-0.1,high=0.1),
        #  pc_transforms.Random_Rotate(),
        #  pc_transforms.Random_Rotate_90(),

        # pc_transforms.Rotate_90(args,axis='x',angle=-1.0),# 1.0,2,3,4
        # pc_transforms.Rotate_90(args, axis='z', angle=2.0),
        # pc_transforms.Rotate_90(args, axis='y', angle=2.0),
        # pc_transforms.Rotate_90(args, axis='shape_complete') TODO this is essential for angela data set
    ])

    input_transforms = transforms.Compose([

        pc_transforms.ArrayToTensor(),
        #   transforms.Normalize(mean=[0.5,0.5],std=[1,1])
    ])

    target_transforms = transforms.Compose([
        pc_transforms.ArrayToTensor(),
        #  transforms.Normalize(mean=[0.5, 0.5], std=[1, 1])
    ])

    """-----------------------------------------------Data Loader----------------------------------------------------"""

    if (args.net_name == 'auto_encoder'):
        [train_dataset, valid_dataset] = Datasets.__dict__[args.dataName](input_root=args.data,
                                                                          target_root=None,
                                                                          split=args.split_value,
                                                                          net_name=args.net_name,
                                                                          input_transforms=input_transforms,
                                                                          target_transforms=target_transforms,
                                                                          co_transforms=co_transforms)
        [test_dataset,_] = Datasets.__dict__[args.dataName](input_root=args.dataIncomplete,
                                                                          target_root=None,
                                                                          split=1.0,
                                                                          net_name=args.net_name,
                                                                          input_transforms=input_transforms,
                                                                          target_transforms=target_transforms,
                                                                          co_transforms=co_transforms)


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=1,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)


    """----------------Model Settings-----------------------------------------------"""

    print('Encoder Model: {0}, Decoder Model : {1}'.format(args.model_encoder,args.model_decoder))
    print('GAN Model Generator:{0} & Discriminator : {1} '.format(args.model_generator,args.model_discriminator))


    network_data_AE = torch.load(args.pretrained_enc_dec)

    network_data_G = torch.load(args.pretrained_G)

    network_data_D = torch.load(args.pretrained_D)

    model_encoder = models.__dict__[args.model_encoder](args, num_points=2048, global_feat=True,
                                                        data=network_data_AE, calc_loss=False).cuda()
    model_decoder = models.__dict__[args.model_decoder](args, data=network_data_AE).cuda()

    model_G = models.__dict__[args.model_generator](args, data=network_data_G).cuda()

    model_D = models.__dict__[args.model_discriminator](args, data=network_data_D).cuda()



    params = get_n_params(model_encoder)
    print('| Number of Encoder parameters [' + str(params) + ']...')

    params = get_n_params(model_decoder)
    print('| Number of Decoder parameters [' + str(params) + ']...')



    chamfer = ChamferLoss(args)
    nll = NLL()
    mse = MSE(reduction = 'elementwise_mean')
    norm = Norm(dims=args.z_dim)

    epoch = 0


    test_loss = trainRL(train_loader, valid_loader,test_loader, model_encoder, model_decoder, model_G,model_D, epoch, args, chamfer,nll, mse, norm, vis_Valid,
                     vis_Valida)
    print('Average Loss :{}'.format(test_loss))


def trainRL(train_loader,valid_loader,test_loader,model_encoder,model_decoder, model_G,model_D,epoch,args, chamfer,nll, mse,norm,vis_Valid,vis_Valida):

    model_encoder.eval()
    model_decoder.eval()
    model_G.eval()
    model_D.eval()

    epoch_size = len(valid_loader)


    file_name = "%s_%s" % (args.policy_name, args.env_name)

    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env = envs(args, model_G, model_D, model_encoder, model_decoder, epoch_size)

    state_dim = args.state_dim
    action_dim = args.z_dim
    max_action = args.max_action

    # Initialize policy
    if args.policy_name == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action)
    elif args.policy_name == "OurDDPG":
        policy = OurDDPG.DDPG(state_dim, action_dim, max_action)
    elif args.policy_name == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action, args.device)

    replay_buffer = utils.ReplayBuffer()

    evaluations = [evaluate_policy(policy,valid_loader,env,args)]



    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True
    env.reset(epoch_size=len(train_loader))


    while total_timesteps < args.max_timesteps:


        if done:

            try:
                input = next(dataloader_iterator)
            except:
                dataloader_iterator = iter(train_loader)
                input = next(dataloader_iterator)


            if total_timesteps != 0:
                # print("Total T: %d Episode Num: %d Episode T: %d Reward: %f") % (total_timesteps, episode_num, episode_timesteps, episode_reward)
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau,
                                 args.policy_noise, args.noise_clip, args.policy_freq)
                else:
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount, args.tau)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq

                evaluations.append(evaluate_policy(policy,valid_loader,env,args,render = False))

                if args.save_models: policy.save(file_name, directory="./pytorch_models")

                env.reset(epoch_size=len(test_loader))
                test_policy(policy, test_loader, env, args, render=True)

                env.reset(epoch_size=len(train_loader))


            # Reset environment
           # obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        obs = env.agent_input(input)

        if total_timesteps < args.start_timesteps:
          #  action_t = torch.rand(args.batch_size, args.z_dim) # TODO checked rand instead of randn
            action_t = torch.FloatTensor(args.batch_size, args.z_dim).uniform_(-args.max_action, args.max_action)
            action = action_t.detach().cpu().numpy().squeeze(0)



           # obs, _, _, _, _ = env(input, action_t)
        else:

           # action_rand = torch.randn(args.batch_size, args.z_dim)
           #
           # obs, _, _, _, _ = env( input, action_rand)

            action = policy.select_action(np.array(obs))
            if args.expl_noise != 0:
                action = (action + np.random.normal(0, args.expl_noise, size=args.z_dim)).clip(
                    -args.max_action*np.ones(args.z_dim,), args.max_action*np.ones(args.z_dim,))
                action = np.float32(action)
            action_t = torch.tensor(action).cuda().unsqueeze(dim=0)
        # Perform action

        # env.render()

        new_obs, _, reward, done, _ = env(input, action_t,disp = True)

       # new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == args.max_episodes_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


    # for i,(input) in enumerate(valid_loader):
    #
    #
    #
    #      if np.shape(input)[0]< args.batch_size:
    #         break;#print(np.shape(input)[0])
    #
    #      action = torch.randn(args.batch_size, args.z_dim)
    #      action_np = action.detach().cpu().numpy()
    #      new_state, _, reward,done, _ = env1(i,input,action)
    #
    #
    #
    # return reward

class envs(nn.Module):
    def __init__(self,args,model_G,model_D,model_encoder,model_decoder,epoch_size):
        super(envs,self).__init__()

        self.nll = NLL()
        self.mse = MSE(reduction='elementwise_mean')
        self.norm = Norm(dims=args.z_dim)
        self.chamfer = ChamferLoss(args)
        self.epoch = 0
        self.epoch_size =epoch_size

        self.model_G = model_G
        self.model_D = model_D
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.j = 1
        self.figures = 3
        self.attempts = args.attempts
        self.end = time.time()
        self.batch_time = AverageMeter()
        self.lossess = AverageMeter()
        self.attempt_id =0
        self.state_prev = np.zeros([4,])
        self.iter = 0
    def reset(self,epoch_size,figures =3):
        self.j = 1;
        self.i = 0;
        self.figures = figures;
        self.epoch_size= epoch_size
    def agent_input(self,input):
        with torch.no_grad():
            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)
            encoder_out = self.model_encoder(input_var, )
            out = encoder_out.detach().cpu().numpy().squeeze()
        return out
    def forward(self,input,action,render=False, disp=False):
        with torch.no_grad():
            # Encoder Input
            input = input.cuda(async=True)
            input_var = Variable(input, requires_grad=True)

            # Encoder  output
            encoder_out = self.model_encoder(input_var, )

            # D Decoder Output
#            pc_1, pc_2, pc_3 = self.model_decoder(encoder_out)
            pc_1 = self.model_decoder(encoder_out)
            # Generator Input
            z = Variable(action, requires_grad=True).cuda()

            # Generator Output
            out_GD, _ = self.model_G(z)
            out_G = torch.squeeze(out_GD, dim=1)
            out_G = out_G.contiguous().view(-1, args.state_dim)

            # Discriminator Output
            #out_D, _ = self.model_D(encoder_out.view(-1,1,32,32))
        #    out_D, _ = self.model_D(encoder_out.view(-1, 1, 1,args.state_dim)) # TODO Alert major mistake
            out_D, _ = self.model_D(out_GD) # TODO Alert major mistake

            # H Decoder Output
#            pc_1_G, pc_2_G, pc_3_G = self.model_decoder(out_G)
            pc_1_G = self.model_decoder(out_G)


            # Preprocesing of Input PC and Predicted PC for Visdom
            trans_input = torch.squeeze(input_var, dim=1)
            trans_input = torch.transpose(trans_input, 1, 2)
            trans_input_temp = trans_input[0, :, :]
            pc_1_temp = pc_1[0, :, :] # D Decoder PC
            pc_1_G_temp = pc_1_G[0, :, :] # H Decoder PC


        # Discriminator Loss
        loss_D = self.nll(out_D)

        # Loss Between Noisy GFV and Clean GFV
        loss_GFV = self.mse(out_G, encoder_out)

        # Norm Loss
        loss_norm = self.norm(z)

        # Chamfer loss
        loss_chamfer = self.chamfer(pc_1_G, pc_1)  # #self.chamfer(pc_1_G, trans_input) instantaneous loss of batch items

        # States Formulation
        state_curr = np.array([loss_D.cpu().data.numpy(), loss_GFV.cpu().data.numpy()
                                  , loss_chamfer.cpu().data.numpy(), loss_norm.cpu().data.numpy()])
      #  state_prev = self.state_prev

        reward_D = state_curr[0]#state_curr[0] - self.state_prev[0]
        reward_GFV =-state_curr[1]# -state_curr[1] + self.state_prev[1]
        reward_chamfer = -state_curr[2]#-state_curr[2] + self.state_prev[2]
        reward_norm =-state_curr[3] # - state_curr[3] + self.state_prev[3]
        # Reward Formulation
        reward = ( reward_D *0.01 + reward_GFV * 10.0 + reward_chamfer *100.0 + reward_norm*1/10)#( reward_D + reward_GFV * 10.0 + reward_chamfer *100 + reward_norm*0.002) #reward_GFV + reward_chamfer + reward_D * (1/30)  TODO reward_D *0.002 + reward_GFV * 10.0 + reward_chamfer *100 + reward_norm  ( reward_D *0.2 + reward_GFV * 100.0 + reward_chamfer *100 + reward_norm)
      #  reward = reward * 100
     #   self.state_prev = state_curr

        #self.lossess.update(loss_chamfer.item(), input.size(0))  # loss and batch size as input

        # measured elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()

       # if i % args.print_freq == 0 :

      #  if self.j <= 5:
        visuals = OrderedDict(
            [('Input_pc', trans_input_temp.detach().cpu().numpy()),
             ('AE Predicted_pc', pc_1_temp.detach().cpu().numpy()),
             ('GAN Generated_pc', pc_1_G_temp.detach().cpu().numpy())])
        if render==True and self.j <= self.figures:
         vis_Valida[self.j].display_current_results(visuals, self.epoch, self.i)
         self.j += 1

        if disp:
            print('[{4}][{0}/{1}]\t Reward: {2}\t States: {3}'.format(self.i, self.epoch_size,reward,state_curr,self.iter))
            self.i += 1
            if(self.i>=self.epoch_size):
                self.i=0
                self.iter +=1


      #  errors = OrderedDict([('loss', loss_chamfer.item())])  # plotting average loss
     #   vis_Valid.plot_current_errors(self.epoch, float(i) / self.epoch_size, args, errors)
        # if self.attempt_id ==self.attempts:
        #     done = True
        # else :
        #     done = False
        done = True
        state = out_G.detach().cpu().data.numpy().squeeze()
        return state, _, reward, done, self.lossess.avg



if __name__ == '__main__':
    args = get_parameters()
    args.device = torch.device(
        "cuda:%d" % (args.gpu_id) if torch.cuda.is_available() else "cpu")  # for selecting device for chamfer loss

    torch.cuda.set_device(args.gpu_id)
    print('Using TITAN XP GPU # :', torch.cuda.current_device())

    print(args)

    """-------------------------------------------------Visualer Initialization-------------------------------------"""

    visualizer = Visualizer(args)

    args.display_id = args.display_id + 10
    args.name = 'Validation'
    vis_Valid = Visualizer(args)
    vis_Valida = []
    args.display_id = args.display_id + 10

    for i in range(1, 15):
        vis_Valida.append(Visualizer(args))
        args.display_id = args.display_id + 10


    main(args,vis_Valid,vis_Valida)







