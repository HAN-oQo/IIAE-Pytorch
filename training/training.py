from models.encoder import Common_Shared_Encoder, Exclusive_Shared_Encoder, Exclusive_Specific_Encoder, Shared_Feature_extractor
from models.decoder import Decoder
from dataloader.dataloader import load_dataloader
import os
import json
import numpy as np
import time
import datetime
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from training.logger import Logger


class Trainer():

    def __init__(self, device, train, directory, path_to_data, batch_size, size, max_iters, resume_iters, restored_model_path, lr, weight_decay, beta1, beta2, milestones, scheduler_gamma, rec_weight, X_kld_weight, S_kld_weight, inter_kld_weight, print_freq, sample_freq, model_save_freq, test_iters, test_path):

        self.device = device
        self.train_bool = train
        ##############
        # Directory Setting
        ###############
        self.directory = directory
        log_dir = os.path.join(directory, "logs")
        sample_dir = os.path.join(directory, "samples")
        result_dir = os.path.join(directory, "results")
        model_save_dir = os.path.join(directory, "models")

        if not os.path.exists(os.path.join(directory, "logs")):
            os.makedirs(log_dir)
        self.log_dir = log_dir

        if not os.path.exists(os.path.join(directory, "samples")):
            os.makedirs(sample_dir)
        self.sample_dir = sample_dir

        if not os.path.exists(os.path.join(directory, "results")):
            os.makedirs(result_dir)
        self.result_dir = result_dir

        if not os.path.exists(os.path.join(directory, "models")):
            os.makedirs(model_save_dir)
        self.model_save_dir = model_save_dir

        ##################
        # Data Loader
        ##################
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.size = size
        self.data_loader , data_length = load_dataloader(dataset = "AFHQ", 
                                                        path_to_data = self.path_to_data,
                                                        train = self.train_bool,
                                                        size = self.size,
                                                        batch_size= self.batch_size
                                                        )

        #################
        # Iteration Setting
        ################
        self.max_iters = max_iters
        self.resume_iters = resume_iters
        self.global_iters = self.resume_iters
        self.restored_model_path = restored_model_path
        

        ##################
        # Optimizer, Scheduler setting
        ###############
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.milestones = milestones
        self.scheduler_gamma = scheduler_gamma

        #################
        # Loss hyperparameters 
        ################
        self.rec_weight = rec_weight
        self.kld_weight = self.batch_size / data_length
        self.X_kld_weight = X_kld_weight
        self.S_kld_weight = S_kld_weight
        self.inter_kld_weight = inter_kld_weight
        
        #################
        # Log Setting
        #################
        self.print_freq = print_freq
        self.sample_freq = sample_freq
        self.model_save_freq = model_save_freq

        #################
        # Constant Tensor
        ##################
        self.normal_mu = torch.tensor(np.float32(0)).to(self.device)
        self.normal_log_var = torch.tensor(np.float32(0)).to(self.device)

        ################
        # Test Setting
        ################
        self.test_iters = test_iters
        self.test_path = test_path
    
        self.build_model()
        self.build_tensorboard()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    
    def build_model(self):
        self.zx_encoder = Exclusive_Specific_Encoder().to(self.device)
        self.zy_encoder = Exclusive_Specific_Encoder().to(self.device)
        
        self.FE = Shared_Feature_extractor().to(self.device)
        self.zs_encoder = Common_Shared_Encoder().to(self.device)
        self.zx_s_encoder = Exclusive_Shared_Encoder().to(self.device)
        self.zy_s_encoder = Exclusive_Shared_Encoder().to(self.device)

        self.x_decoder = Decoder().to(self.device)
        self.y_decoder = Decoder().to(self.device)
        
        self.optimizer_exc_enc= torch.optim.Adam(itertools.chain(self.zx_encoder.parameters(), self.zy_encoder.parameters()), lr = self.lr, betas = (self.beta1, self.beta2), weight_decay = self.weight_decay)
        self.optimizer_shr_enc= torch.optim.Adam(itertools.chain(self.FE.parameters(), self.zs_encoder.parameters(),self.zx_s_encoder.parameters(), self.zy_s_encoder.parameters()), 
                                                lr = self.lr, betas = (self.beta1, self.beta2), weight_decay = self.weight_decay)
        self.optimizer_dec = torch.optim.Adam(itertools.chain(self.x_decoder.parameters(), self.y_decoder.parameters()), lr = self.lr, betas = (self.beta1, self.beta2), weight_decay = self.weight_decay)

        self.scheduler_exc_enc = torch.optim.lr_scheduler.MultiStepLR(optimizer = self.optimizer_exc_enc, milestones = self.milestones, gamma = self.scheduler_gamma)
        self.scheduler_shr_enc = torch.optim.lr_scheduler.MultiStepLR(optimizer = self.optimizer_shr_enc, milestones = self.milestones, gamma = self.scheduler_gamma)
        self.scheduler_dec = torch.optim.lr_scheduler.MultiStepLR(optimizer = self.optimizer_dec, milestones = self.milestones, gamma = self.scheduler_gamma)

        # self.print_network(self.zx_encoder, 'ZX_Encoder')
        # self.print_network(self.zy_encoder, 'ZY_Encoder')
        # self.print_network(self.FE, 'Feature Extractor')
        # self.print_network(self.zx_s_encoder, 'ZX_S_Encoder')
        # self.print_network(self.zy_s_encoder, 'ZY_S_Encoder')
        # self.print_network(self.zs_encoder, 'ZS_Encoder')
        # self.print_network(self.decoder, 'decoders')
        

    def load_model(self, path, rersume_iters):
        """Restore the trained generator and discriminator."""
        resume_iters = int(resume_iters)
        print('Loading the trained models from iters {}...'.format(resume_iters))
        path = os.path.join( path , '{}-checkpoint.pt'.format(resume_iters))
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        self.global_iters = checkpoint['iters']
        self.zx_encoder.load_state_dict(checkpoint['zx_encoder'])
        self.zy_encoder.load_state_dict(checkpoint['zy_encoder'])
        self.FE.load_state_dict(checkpoint['FE'])
        self.zx_s_encoder.load_state_dict(checkpoint['zx_s_encoder'])
        self.zy_s_encoder.load_state_dict(checkpoint['zy_s_encoder'])
        self.zs_encoder.load_state_dict(checkpoint['zs_encoder'])
        self.x_decoder.load_state_dict(checkpoint['x_decoder'])
        self.y_decoder.load_state_dict(checkpoint['y_decoder'])
        self.optimizer_exc_enc.load_state_dict(checkpoint['optimizer_exc_enc'])
        self.optimizer_shr_enc.load_state_dict(checkpoint['optimizer_shr_enc'])
        self.optimizer_dec.load_state_dict(checkpoint['optimizer_dec'])
        self.scheduler_exc_enc.load_state_dict(checkpoint['scheduler_exc_enc'])
        self.scheduler_shr_enc.load_state_dict(checkpoint['scheduler_shr_enc'])
        self.scheduler_dec.load_state_dict(checkpoint['scheduler_dec'])
        # self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage)
        
    
    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.log_dir)

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    def reconstruction_loss(self, recon, input):
        rec_loss = nn.L1Loss()
        return rec_loss(recon, input)
    
    def KLD_loss_v1(self, mu, log_var):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss
    
    def KLD_loss(self, mu0, log_var0, mu1, log_var1):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + (log_var0 - log_var1) - ((mu0 - mu1) ** 2 + log_var0.exp()/log_var1.exp()), dim = 1), dim = 0)
        return self.kld_weight * kld_loss

    def loss_function(self, recon_X, recon_Y, input_X, input_Y, zx_mu, zy_mu, zx_s_mu, zy_s_mu, zs_mu, zx_log_var, zy_log_var, zx_s_log_var, zy_s_log_var, zs_log_var):

        recon_X_loss = self.reconstruction_loss(recon_X, input_X)
        recon_Y_loss = self.reconstruction_loss(recon_Y, input_Y)

        

        kl_X_loss = self.KLD_loss(zx_mu, zx_log_var, self.normal_mu, self.normal_log_var)
        kl_Y_loss = self.KLD_loss(zy_mu, zy_log_var, self.normal_mu, self.normal_log_var)
        kl_S_loss = self.KLD_loss(zs_mu, zs_log_var, self.normal_mu, self.normal_log_var)
        kl_interX_loss = self.KLD_loss(zs_mu, zs_log_var, zx_s_mu, zx_s_log_var)
        kl_interY_loss = self.KLD_loss(zs_mu, zs_log_var, zy_s_mu, zy_s_log_var)

        reg_coeff = (1.0 - ((-1)*torch.tensor(self.global_iters, dtype = torch.float32)/25000.0).exp()).to(self.device)

        total_loss = self.rec_weight * (recon_X_loss + recon_Y_loss) \
                    + reg_coeff * self.X_kld_weight * (kl_X_loss + kl_Y_loss) \
                    + reg_coeff * self.S_kld_weight * (kl_S_loss) \
                    + self.inter_kld_weight * (kl_interX_loss + kl_interY_loss)

        return total_loss , [recon_X_loss.item(), recon_Y_loss.item(), kl_X_loss.item(), kl_Y_loss.item(), kl_S_loss.item(), kl_interX_loss.item(), kl_interY_loss.item()]

    def Unpack_Data(self, data):
        cat_image = data["Cat"].detach().float().to(self.device)
        dog_image = data["Dog"].detach().float().to(self.device)
        return cat_image, dog_image
    
    def train(self):
        
        data_iter = iter(self.data_loader)
        data_fixed = next(data_iter)
        X_fixed , Y_fixed = self.Unpack_Data(data_fixed)

        if self.resume_iters > 0:
            self.load_model(self.restored_model_path, self.resume_iters)
            self.zx_encoder.to(self.device)
            self.zy_encoder.to(self.device)
            self.FE.to(self.device)
            self.zs_encoder.to(self.device)
            self.zx_s_encoder.to(self.device)
            self.zy_s_encoder.to(self.device)
            self.x_decoder.to(self.device)
            self.y_decoder.to(self.device)


        print("Start Training...")
        start_time = time.time()
        while self.global_iters <= self.max_iters:
            try:
                input_X, input_Y = self.Unpack_Data(next(data_iter))
            except:
                data_iter = iter(self.data_loader)
                input_X, input_Y = self.Unpack_Data(next(data_iter))

            self.global_iters += 1

            self.optimizer_exc_enc.zero_grad()
            self.optimizer_shr_enc.zero_grad()
            self.optimizer_dec.zero_grad()

            zx_mu, zx_log_var, zx = self.zx_encoder(input_X)
            zy_mu, zy_log_var, zy = self.zy_encoder(input_Y)
            feature_X = self.FE(input_X)
            feature_Y = self.FE(input_Y)
            zx_s_mu, zx_s_log_var, zx_s = self.zx_s_encoder(feature_X)
            zy_s_mu, zy_s_log_var, zy_s = self.zy_s_encoder(feature_Y)
            zs_mu, zs_log_var, zs = self.zs_encoder(feature_X, feature_Y)

            recon_X = self.x_decoder(zx, zs)
            recon_Y = self.y_decoder(zy, zs)

            loss, item = self.loss_function(recon_X, recon_Y, input_X, input_Y, 
                                    zx_mu, zy_mu, zx_s_mu, zy_s_mu, zs_mu, 
                                    zx_log_var, zy_log_var, zx_s_log_var, zy_s_log_var, zs_log_var)
            

            loss.backward()
            self.optimizer_exc_enc.step()
            self.optimizer_shr_enc.step()
            self.optimizer_dec.step()

            
            loss_item = {}
            loss_item["recon_X_loss"] = item[0]
            loss_item["recon_Y_loss"] = item[1]
            loss_item["kl_X_loss"] = item[2]
            loss_item["kl_Y_loss"] = item[3]
            loss_item["kl_S_loss"] = item[4]
            loss_item["kl_interX_loss"] = item[5]
            loss_item["kl_interY_loss"] = item[6]
            loss_item["total_loss"] = loss.item()

            if self.global_iters % self.print_freq == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration[{}/{}]".format(et, self.global_iters, self.max_iters)
                for tag, value in loss_item.items():
                    log += ", {}: {:4f}".format(tag, value)
                print(log)

                for tag, value in loss_item.items():
                    self.logger.scalar_summary(tag, value, self.global_iters)
            

            if self.global_iters % self.sample_freq == 0:
                with torch.no_grad():
                    ith_sample_dir = os.path.join(self.sample_dir, str(self.global_iters))
                    if not os.path.exists(ith_sample_dir):
                        os.makedirs(ith_sample_dir)

                    _, _, z_x = self.zx_encoder(X_fixed)
                    _, _, z_y = self.zy_encoder(Y_fixed)
                    fx = self.FE(X_fixed)
                    fy = self.FE(Y_fixed)
                    _, _ , zx_s = self.zx_s_encoder(fx)
                    _, _ , zy_s = self.zy_s_encoder(fy)
                    _, _ , z_s = self.zs_encoder(fx, fy)

                    X_fake_list = [X_fixed]
                    Y_fake_list = [Y_fixed]

                    recon_X = self.x_decoder(z_x, z_s)
                    Y2X = self.x_decoder(z_x, zy_s)
                    rand0_Y2X= self.x_decoder(torch.randn_like(z_x), zy_s)
                    rand1_Y2X= self.x_decoder(torch.randn_like(z_x), zy_s)
                    rand2_Y2X= self.x_decoder(torch.randn_like(z_x), zy_s)
                    X_fake_list.append(recon_X)
                    X_fake_list.append(Y2X)
                    X_fake_list.append(rand0_Y2X)
                    X_fake_list.append(rand1_Y2X)
                    X_fake_list.append(rand2_Y2X)
                    X_fake_list.append(Y_fixed)

                    recon_Y = self.y_decoder(z_y, z_s)
                    X2Y =  self.y_decoder(z_y, zx_s)
                    rand0_X2Y = self.y_decoder(torch.randn_like(z_y), zx_s)
                    rand1_X2Y = self.y_decoder(torch.randn_like(z_y), zx_s)
                    rand2_X2Y = self.y_decoder(torch.randn_like(z_y), zx_s)
                    Y_fake_list.append(recon_Y)
                    Y_fake_list.append(X2Y)                
                    Y_fake_list.append(rand0_X2Y)                
                    Y_fake_list.append(rand1_X2Y)                
                    Y_fake_list.append(rand2_X2Y)  
                    Y_fake_list.append(X_fixed)              
                    
                    X_concat = torch.cat(X_fake_list, dim=3)
                    Y_concat = torch.cat(Y_fake_list, dim=3)
                    sampleX_path = os.path.join(ith_sample_dir, "Dog2Cat.jpg")
                    sampleY_path = os.path.join(ith_sample_dir, "Cat2Dog.jpg")

                    save_image(self.denorm(X_concat.cpu()), sampleX_path, nrow=1, padding =0)
                    save_image(self.denorm(Y_concat.cpu()), sampleY_path, nrow=1, padding =0)
                    print('Saved samples into {}...'.format(ith_sample_dir))

            if self.global_iters % self.model_save_freq == 0:
                    model_path = os.path.join(self.model_save_dir, "{}-checkpoint.pt".format(self.global_iters))
                    torch.save({
                        'iters': self.global_iters,
                        'zx_encoder': self.zx_encoder.state_dict(),
                        'zy_encoder': self.zy_encoder.state_dict(),
                        'FE': self.FE.state_dict(),
                        'zs_encoder': self.zs_encoder.state_dict(),
                        'zx_s_encoder': self.zx_s_encoder.state_dict(),
                        'zy_s_encoder': self.zy_s_encoder.state_dict(),
                        'x_decoder': self.x_decoder.state_dict(),
                        'y_decoder': self.y_decoder.state_dict(),
                        'optimizer_exc_enc': self.optimizer_exc_enc.state_dict(),
                        'optimizer_shr_enc': self.optimizer_shr_enc.state_dict(),
                        'optimizer_dec': self.optimizer_dec.state_dict(),
                        'scheduler_exc_enc': self.scheduler_exc_enc.state_dict(),
                        'scheduler_shr_enc':self.scheduler_shr_enc.state_dict(),
                        'scheduler_dec': self.scheduler_dec.state_dict()    
                    }, model_path)
                    # torch.save(self.model.state_dict(), model_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))
            

    def test(self):
        return


    