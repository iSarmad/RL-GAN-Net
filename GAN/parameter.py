import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Data loader settings
    #parser.add_argument('-d', '--data', metavar='DIR',default='/home/sarmad/PycharmProjects/pointShapeComplete/GFV/shapenet/08-16-16:11/encoder_pointnet', help='Path to Complete Point Cloud Data Set')
   # parser.add_argument('-d', '--data', metavar='DIR',default='/home/sarmad/PycharmProjects/pointShapeComplete/GFV/shapenet/09-05-15:10/encoder_pointnet', help='Path to Complete Point Cloud Data Set')
    # TODO Path to the GFV data
    parser.add_argument('-d', '--data', metavar='DIR',default=    '', help='Path to Complete Point Cloud Data Set')



    parser.add_argument('-s', '--split_value', default=0.9, help='Ratio of train and test data split')
    parser.add_argument('-n', '--dataName', metavar='Data Set Name', default='shapenet')

 # TODO add path to pretrained model
    parser.add_argument('--pretrained',
                        default='')
    # parser.add_argument('--pretrained',
    #                     default='/home/sarmad/PycharmProjects/pointShapeComplete/ckpts/shapenet/09-12-21:00/ae_pointnet,Adam,1000epochs,b24,lr0.001/model_best.pth.tar',
    #                     help='Use Pretrained Model for testing or resuming training')  ## TODO

    parser.add_argument('-me', '--model_encoder', default='encoder_pointnet', help='Chose Your Encoder Model Here',
                        choices=['encoder_pointnet'])  # TODO
    parser.add_argument('-md', '--model_decoder', default='decoder_sonet', help='Chose Your Decoder Model Here',
                        choices=['decoder_sonet'])  # TODO

    # Setting for Decoder
    # parser.add_argument('--output_pc_num', type=int, default=1280, help='# of output points')
    parser.add_argument('--output_fc_pc_num', type=int, default=256, help='# of fc decoder output points')
    parser.add_argument('--output_conv_pc_num', type=int, default=4096, help='# of conv decoder output points')
    parser.add_argument('--feature_num', type=int, default=1024, help='length of encoded feature')
    parser.add_argument('--activation', type=str, default='relu', help='activation function: relu, elu')
    parser.add_argument('--normalization', type=str, default='batch', help='normalization function: batch, instance')

    # Visualizer Settings
    parser.add_argument('--name', type=str, default='GFV',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--display_id', type=int, default=2000, help='window id of the web display')
    parser.add_argument('--port_id', type=int, default=8099, help='Port id for browser') #TODO This
    parser.add_argument('--print_freq', type=int, default=10, help='Print Frequency')

    # GPU settings
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu ids: e.g. 0, 1. -1 is no GPU') #TODO This

    # Model hyper-parameters
    parser.add_argument('--max_action', type=float, default=10)
    parser.add_argument('--model', type=str, default='sagan', choices=['sagan', 'qgan'])
    parser.add_argument('--adv_loss',default='wgan-gp', type=str, choices=['wgan-gp', 'hinge']) #
    parser.add_argument('--imsize',default=32, type=int) #
    parser.add_argument('--imsize_new', default=32, type=int)
    parser.add_argument('--g_num', type=int, default=5)
    parser.add_argument('--z_dim', type=int, default=1)#
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10) # gradient penalty
    parser.add_argument('--version', default='sagan_celeb',type=str)

    # Training setting
    parser.add_argument('--total_step', type=int, default=1000000, help='how many times to update the generator')
    parser.add_argument('--d_iters', type=float, default=5)
    parser.add_argument('--batch_size', default = 50, type=int) #
    parser.add_argument('--num_workers', type=int, default=2) #
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)#
    parser.add_argument('--lr_decay', type=float, default=0.0)#
    parser.add_argument('--beta1', type=float, default=0.5)#
    parser.add_argument('--beta2', type=float, default=0.9)#

    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--dataset', default = 'celeb', type=str,choices=['lsun', 'celeb'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')
    parser.add_argument('--attn_path', type=str, default='./attn')

    # Step size
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=float, default=10.0)




    return parser.parse_args()