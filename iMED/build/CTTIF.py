import gc
import os
import time

import torch
import torch.nn as nn

from iMED.utils.load import *
from iMED.utils.utils import *
from iMED.utils.save import *
from torch.autograd import Variable
from pytorch_ssim import *
from tensorboardX import SummaryWriter


class CTTIF:
    def __init__(self, cfg_path):
        super().__init__()
        self.cfg = load_config(cfg_path)

        if self.cfg['project']['train']:
            seed = None
            if self.cfg['data']['seed'] != 'None':
                seed = self.cfg['data']['seed']
                setup_seed(self.cfg['data']['seed'])

            self.train_loader, self.test_loader, input_size = load_train_val_dataloader(
                file_path=self.cfg['data']['file_path'],
                input_height=self.cfg['data']['input_height'],
                input_width=self.cfg['data']['input_weight'],
                use_oct=self.cfg['data']['use_oct'],
                batch_size=self.cfg['train']['batch_size'],
                test_size=self.cfg['data']['test_size'],
                shuffle=self.cfg['data']['shuffle'],
                seed=seed
            )
            self.model = load_model(
                c1=self.cfg['model']['c'],
                c2=self.cfg['model']['c_out'],
                n=self.cfg['model']['n'],
                input_size=input_size,
                is_new=self.cfg['model']['new_generator']
            )
        else:
            self.model = load_model(
                c1=self.cfg['model']['c'],
                c2=self.cfg['model']['c_out'],
                n=self.cfg['model']['n'],
                input_size=None,
                is_new=self.cfg['model']['new_generator']
            )

        if self.cfg['model']['pth'] != 'None':
            self.model.load_state_dict(torch.load(self.cfg['model']['pth']))

        self.save_path = generate_folder(
            self.cfg['project']['name'],
            self.cfg['project']['save_dir'],
            self.cfg['project']['train']
        )

    def train(self):
        config_path = os.path.join(self.save_path, 'config.yaml')
        with open(config_path, "w") as f:
            yaml.dump(self.cfg, f)

        device = torch.device('cpu')
        if self.cfg['train']['cuda'] != 'None':
            if torch.cuda.is_available():
                device = torch.device("cuda:{}".format(self.cfg['train']['cuda'][0]))

        statistical_iteration = 1
        channel = self.cfg['model']['c_out'][0] if not isinstance(self.cfg['model']['c_out'], int) else \
            self.cfg['model']['c_out']
        best_val_loss = 1e7

        optimizer_G = load_optim(
            self.cfg['train']['optim_G'],
            self.model.generator,
            self.cfg['train']['lr_0_G'],
            self.cfg['train']['optim_G_parameters']
        )
        s_G = load_lr_scheduler(
            optimizer_G, self.cfg['train']['lr_f_G'],
            self.cfg['train']['epochs'],
            self.cfg['train']['step_size']
        )
        optimizer_D = load_optim(
            self.cfg['train']['optim_D'],
            self.model.discriminator,
            self.cfg['train']['lr_0_D'],
            self.cfg['train']['optim_G_parameters']
        )
        s_D = load_lr_scheduler(
            optimizer_D, self.cfg['train']['lr_f_D'],
            self.cfg['train']['epochs'],
            self.cfg['train']['step_size']
        )

        self.model.to(device)
        BME_loss = nn.MSELoss(reduction='mean').to(device)
        ssim_loss = SSIM(window_size=11).to(device)
        adversarial_loss = nn.BCEWithLogitsLoss().to(device)

        log_writer = SummaryWriter(self.save_path)
        for epoch in range(self.cfg['train']['epochs']):
            print("Start training in epoch {}".format(epoch))
            save_path = os.path.join(self.save_path, f'epoch{epoch}')
            s = time.time()
            train_loss = AverageMeter()
            d_loss_avg = 0
            g_loss_avg = 0
            self.model.train()
            for i, (input, target) in enumerate(self.train_loader):
                input_var, target_var = Variable(input), Variable(target)
                input_var = input_var.to(device)
                target_var = target_var.to(device)

                # 训练判别器
                optimizer_D.zero_grad()
                real_outputs = self.model.discriminator(target_var)
                real_loss = adversarial_loss(
                    real_outputs, torch.ones_like(real_outputs).to(device))
                gen_outputs, fake_outputs = self.model(input_var)
                fake_loss = adversarial_loss(
                    fake_outputs, torch.zeros_like(fake_outputs).to(device))
                d_loss = 0.1 * (real_loss + fake_loss)
                d_loss_avg += d_loss.item()
                d_loss.backward(retain_graph=True)
                optimizer_D.step()

                # 训练生成器
                optimizer_G.zero_grad()
                gen_outputs, fake_outputs = self.model(input_var)
                criterion_loss = (1000 * (1 - ssim_loss(gen_outputs, target_var)) + BME_loss(gen_outputs, target_var))
                g_loss = 0.001 * (
                        criterion_loss + adversarial_loss(fake_outputs, torch.ones_like(fake_outputs).to(device)))
                g_loss_avg += g_loss.item()
                g_loss.backward()
                optimizer_G.step()

                # 更新 train_loss
                train_loss.update(g_loss.item() + d_loss.item(), input.size(0))

                if epoch % self.cfg['train']['save_period'] == 0:
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if i % self.cfg['train']['save_iter_per_epoch'] == 0 or i == len(self.train_loader) - 1:
                        print('Epoch: [{0}] \t'
                              'Iter: [{1}/{2}]\t'
                              'Generator Loss: {g_loss_val:.5f}\t'
                              'Discriminator Loss: {d_loss_val:.5f}\t'.format(
                            epoch, i, len(self.train_loader) - 1,
                            g_loss_val=g_loss.data.item(),
                            d_loss_val=d_loss.data.item()))
                        pd = gen_outputs.data.float().cpu()
                        gt = target.data.float().cpu()
                        save_comparison_images(pd, gt, save_dir=save_path, mode="train", epoch=epoch, iter=i)
                torch.cuda.empty_cache()

            log_writer.add_scalar('Train/loss', train_loss.avg, epoch)
            log_writer.add_scalar('Train/d_loss', d_loss_avg / len(self.train_loader), epoch)
            log_writer.add_scalar('Train/g_loss', g_loss_avg / len(self.train_loader), epoch)

            print('Finish Epoch: [{0}]\t'
                  'Average Train Loss: {loss.avg:.5f}\t'.format(
                epoch, loss=train_loss))
            e = time.time()
            print("Time used in training one epoch: ", (e - s))

            self.model.eval()
            with torch.no_grad():
                print("Start validating in epoch {}".format(epoch))
                s = time.time()
                mse, mae, rmse, psnr, ssim = 0, 0, 0, 0, 0
                cnt = 0
                for i, (input, target) in enumerate(self.test_loader):
                    input_var, target_var = Variable(input), Variable(target)
                    input_var = input_var.to(device)
                    output = self.model(input_var)
                    pd = output[0].data.float().cpu()
                    gt = target.data.float().cpu()
                    if epoch % self.cfg['train']['save_period'] == 0:
                        save_comparison_images(pd, gt, save_dir=save_path, mode="validation", epoch=epoch,
                                               iter=i)

                    mse_pd, mae_pd, rmse_pd, psnr_pd, ssim_pd = get_error_metrics(pd, gt)
                    mse += mse_pd
                    mae += mae_pd
                    rmse += rmse_pd
                    psnr += psnr_pd
                    ssim += ssim_pd
                    cnt += 1
                    torch.cuda.empty_cache()

                if epoch % self.cfg['train']['save_period'] == 0:
                    model_path = os.path.join(save_path, "model")
                    os.makedirs(model_path)
                    filename = os.path.join(model_path, "model_val.pth")
                    torch.save(self.model.state_dict(), filename)

                mse /= cnt
                mse /= channel
                mae /= cnt
                mae /= channel
                rmse /= cnt
                rmse /= channel
                psnr /= cnt
                psnr /= channel
                ssim /= cnt
                ssim /= channel
                print('Average mse: {mse_pred:.4f} | mae: {mae_pred:.4f} | rmse: {rmse_pred:.4f} |'
                      ' psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'
                      .format(mse_pred=mse,
                              mae_pred=mae,
                              rmse_pred=rmse,
                              psnr_pred=psnr,
                              ssim_pred=ssim))
                if mse < best_val_loss:
                    best_val_loss = mse
                    best_path = save_path + '_best'
                    best_model_path = os.path.join(best_path, 'model')
                    os.makedirs(best_model_path)
                    filename = os.path.join(best_model_path, "model_val.pth")
                    torch.save(self.model.state_dict(), filename)
                    for i, (input, target) in enumerate(self.test_loader):
                        input_var, target_var = Variable(input), Variable(target)
                        input_var = input_var.to(device)
                        output = self.model(input_var)
                        pd = output[0].data.float().cpu()
                        gt = target.data.float().cpu()
                        save_comparison_images(pd, gt, save_dir=best_path, mode="validation", epoch=epoch,
                                               iter=i)
                    statistical_iteration = 1

                if statistical_iteration >= self.cfg['train']['stop_iter']:
                    print(
                        f'The model cannot be improved anymore, the best model is at {epoch - statistical_iteration + 1} epoch')
                    break
                else:
                    statistical_iteration += 1
                # 将验证集指标写入TensorboardX
                log_writer.add_scalar('Validation/mse', mse, epoch)
                log_writer.add_scalar('Validation/mae', mae, epoch)
                log_writer.add_scalar('Validation/rmse', rmse, epoch)
                log_writer.add_scalar('Validation/psnr', psnr, epoch)
                log_writer.add_scalar('Validation/ssim', ssim, epoch)
                e = time.time()
                print("Time used in validating one epoch: ", (e - s))
                gc.collect()
                s_G.step()
                s_D.step()

    # 在测试函数中添加保存图像的功能
    def test(self, x):
        config_path = os.path.join(self.save_path, 'config.yaml')
        with open(config_path, "w") as f:
            yaml.dump(self.cfg, f)
    
        self.model.to(x.device)
        with torch.no_grad():
            output = self.model(x)
    
        # 创建保存图像的文件夹
        save_images_path = 'iMED/test_images'

        pd = output[0].data.float().cpu()
        
        save_test_images(pd, mode="test", save_dir=save_images_path)
    
        return output



    