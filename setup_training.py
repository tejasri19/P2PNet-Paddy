import os
import random
import datetime
import time
import torch
import numpy as np
from datasets import build_dataset, SHA
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from collections import defaultdict
from typing import Optional, List

from models import build_model
from util.misc import collate_fn_crowd
from torch.utils.tensorboard import SummaryWriter

import math

args = {}

args["row"] = 2
args["line"] = 2
args["point_loss_coef"] = 0.0002
args["set_cost_class"] =1
args["set_cost_point"] =0.05
args["eos_coef"]=0.5
args["dataset_file"]='SHA'
args["backbone"] = 'resnet18'
args["data_root"]='Paddy_dataset'
args["lr"] = 1e-4
args["lr_backbone"] = 1e-5
args["lr_fpn"]=1e-5
args["weight_decay"] = 1e-4
args["gamma"] = 0.1
args["loss_ce"] = 1
args["loss_points"] = 2e-4
args["epochs"] = 300
args["eval_interval"] = 5
args["seed"] = 42
#args["lr_drop"] = 0.1
args["step_size"] = 300
args["resume"] = ''
args["eval"] = True
args["start_epoch"] = 0
args["output_dir"] = '/u/student/2023/ai23mtech02003/yuvraj/Pardhu/paddy_exp/logs'
args["checkpoints_dir"] = '/u/student/2023/ai23mtech02003/yuvraj/Pardhu/paddy_exp/ckpt_resnet_kmo'

class TensorboardLogger:
    def __init__(self, log_dir):
        """
        Initialize TensorBoard writer.
        
        Args:
            log_dir: Directory where TensorBoard logs will be saved
        """
        self.log_dir = os.path.join(log_dir, 'tensorboard')
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
    def log_training(self, loss_ce, loss_points, total_loss, learning_rate, epoch):
        """Log training metrics"""
        self.writer.add_scalar('Training/Classification_Loss', loss_ce, epoch)
        self.writer.add_scalar('Training/Points_Loss', loss_points, epoch)
        self.writer.add_scalar('Training/Total_Loss', total_loss, epoch)
        self.writer.add_scalar('Training/Learning_Rate', learning_rate, epoch)
        
    def log_validation(self, mae, mse, epoch):
        """Log validation metrics"""
        self.writer.add_scalar('Validation/MAE', mae, epoch)
        self.writer.add_scalar('Validation/RMSE', mse, epoch)
        
    def log_histograms(self, model, epoch):
        """Log model parameter distributions"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f'Parameters/{name}', param.data, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()

class DensityBasedSampler(Sampler):
    """
    A sampler that rebalances the data based on point density in images.
    This helps handle long-tailed distributions by oversampling images with fewer points
    and undersampling images with many points.
    """
    def __init__(self, dataset: SHA, 
                num_samples: Optional[int] = None,
                density_power: float = 0.75,
                min_points_weight: float = 0.1):
        """
        Args:
            dataset: The SHA dataset
            num_samples: Number of samples to draw. If None, uses dataset length
            density_power: Power factor to adjust the density weight (smaller values = more aggressive rebalancing)
            min_points_weight: Minimum weight for any sample to prevent extreme undersampling
        """
        self.dataset = dataset
        self.num_samples = len(dataset) if num_samples is None else num_samples
        self.density_power = density_power
        self.min_points_weight = min_points_weight
        
        # Calculate points per image
        self.points_per_image = []
        for img_data in dataset.img_list:
            gt_path = img_data[1]
            with open(gt_path) as f:
                num_points = len(f.readlines())
            self.points_per_image.append(num_points)
            
        self.points_per_image = np.array(self.points_per_image)
        
        # Calculate sampling weights
        weights = self._calculate_weights()
        self.weights = torch.DoubleTensor(weights)
    
    def _calculate_weights(self) -> np.ndarray:
        """Calculate sampling weights based on point density."""
        # Normalize point counts
        max_points = np.max(self.points_per_image)
        normalized_points = self.points_per_image / max_points
        
        # Calculate weights (inverse of density raised to power)
        weights = (1 / (normalized_points + self.min_points_weight)) ** self.density_power
        
        # Normalize weights
        weights = weights / np.sum(weights)
        return weights
    
    def __iter__(self):
        """Return an iterator over indices, weighted by density."""
        rand_tensor = torch.multinomial(
            self.weights, 
            self.num_samples,
            replacement=True
        )
        return iter(rand_tensor.tolist())
    
    def __len__(self):
        return self.num_samples
    
    
def main(args):
    min_metrics = [1e7,1e7]
    # create the logging file
    best_epoch = -1

    tb_logger = TensorboardLogger(args["output_dir"])

    if not os.path.exists(args["output_dir"]):
        os.makedirs(args["output_dir"])
    run_log_name = os.path.join(args["output_dir"], 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("device", device)

    # fix the seed for reproducibility
    seed = args["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion = build_model(args, training=True)
    model.to(device)
    criterion.to(device)
    criterion.train()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
        "weight_decay": args["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args["lr_backbone"],
            "weight_decay": args["weight_decay"],
        },
    ]

    
    optimizer = torch.optim.Adam(param_dicts, lr=args["lr"])
                                #weight_decay=args["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args["step_size"])
#     lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=args["epochs"], eta_min=1e-6
# )

    loading_data = build_dataset(args=args)

    train_set, val_set = loading_data(args["data_root"])
    train_sampler = DensityBasedSampler(train_set)
    train_data_loader = DataLoader(train_set, batch_size=4, sampler = train_sampler, collate_fn=collate_fn_crowd, num_workers= 2,
                            persistent_workers=True,pin_memory=True)
    
    
    val_data_loader = DataLoader(val_set, batch_size=1,collate_fn=collate_fn_crowd, num_workers= 2,
                            persistent_workers=True,pin_memory=True)
    
    # resume the weights and training state if exists
    if args["resume"]:
        checkpoint = torch.load(args["resume"], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        args["start_epoch"] = checkpoint['epoch']
        new_start = 1
        if not args["eval"] and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    else:
        new_start = 0


    
    start_time = time.time()
    
    for epoch in range(args["start_epoch"], args["epochs"]):
        model.train()
        start = time.time()
        epoch_losses_ce = []
        epoch_losses_points = []
        current_lr = optimizer.param_groups[0]['lr']
        
        #losses = []
        
        for idx,(samples, targets) in enumerate(train_data_loader):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(samples)
            loss_dict = criterion(out, targets)
            
            
            epoch_losses_ce.append(loss_dict["loss_ce"].item())
            epoch_losses_points.append(loss_dict["loss_points"].item())
            
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            
        # Calculate average losses for the epoch
        avg_loss_ce = np.mean(epoch_losses_ce)
        avg_loss_points = np.mean(epoch_losses_points)
        total_loss = avg_loss_ce + avg_loss_points
        
        # Log to TensorBoard
        tb_logger.log_training(
            loss_ce=avg_loss_ce,
            loss_points=avg_loss_points,
            total_loss=total_loss,
            learning_rate=current_lr,
            epoch=epoch
        )
    
        # Log parameter distributions every 5 epochs
        if epoch % 5 == 0:
            tb_logger.log_histograms(model, epoch)
        
        stop = time.time()
        train_duration = stop-start
        #losses = [sum(x)/len(x) for x in losses]
        
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], train_duration))
        
        
    

        # with open(run_log_name, "a") as log_file:
        #     log_file.write('[ep %d][lr %.7f][%.2fs]\n' % (epoch, current_lr, train_duration))
        #     log_file.write("classification_loss: {0:.4f}, regression_loss: {1:.4f}\n".format(avg_loss_ce, avg_loss_points))
        #     log_file.flush()
            
        # print("epoch : {0} | lr: {1:.7f} | train_duration : {2:.2f}s | classification_loss : {3:.4f} | regression_loss : {4:.4f}".format(
        #     epoch, current_lr, train_duration, avg_loss_ce, avg_loss_points))
            
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(args["checkpoints_dir"], 'latest.pth')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }, checkpoint_latest_path)
        
        #print("------------------------------------Training------------------------------------------------------------")
        print("epoch : {0} | train_duration : {1} | classification_loss : {2} | regression_loss : {3}".format(epoch,train_duration,avg_loss_ce, avg_loss_points))

        
        
        #evaluation
        if (epoch +1) % args["eval_interval"] ==0 and epoch > 0:
            model.eval()
            maes = []
            mses = []
            val_start = time.time()

            with torch.no_grad():
    
                for samples, targets in val_data_loader:
                    samples = samples.to(device)
                    outputs = model(samples)
                    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
                    outputs_points = outputs['pred_points'][0]
                    points = outputs_points[outputs_scores > 0.5].detach().cpu().numpy()
                    pred_cnt = points.shape[0]
                    gt_cnt = targets[0]['point'].shape[0]
                    
                    
                    mae = abs(pred_cnt-gt_cnt)
                    mse = (pred_cnt - gt_cnt) ** 2
                        
                    maes.append(mae)
                    mses.append(mse)
                        
            val_duration = time.time() - val_start
            
            # calculate the mean mae and mse
            result_mae = np.mean(maes)
            result_rmse = np.sqrt(np.mean(mses))
            
            #Save the best model
            if result_mae < min_metrics[0]:
                min_metrics[0] = result_mae
                min_metrics[1] = result_rmse
                best_epoch = epoch
                checkpoint_path = os.path.join(args["checkpoints_dir"], 'best_mae.pth')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'best_mae': min_metrics[0],
                    'best_epoch': best_epoch
                }, checkpoint_path)
                print(f"Best model saved at epoch {best_epoch} with MAE: {min_metrics[0]}")
                
            print("------------------------------------Testing------------------------------------------------------------")
            print("val_duration : {0} | mae : {1} | mse : {2} | best_mae : {3}".format(val_duration,result_mae,result_rmse,min_metrics[0]))
            print("-------------------------------------------------------------------------------------------------------")
            print(f"Best MAE: {min_metrics[0]:.2f} at epoch {best_epoch}")
            
            #logging
            with open(run_log_name, "a") as log_file:
                # log_file.write(f"mae: {mae}, mse: {mse}, time: {val_duration}, best mae: {min_metrics[0]}\n")
                # log_file.write("Best MAE: {0} at epoch {1}\n".format(min_metrics[0], best_epoch))
                log_file.write(f"Epoch {epoch}: mae={result_mae:.2f}, rmse={result_rmse:.2f}, "
                        f"time={val_duration:.2f}s, best_mae={min_metrics[0]:.2f}\n")
                

        # Update learning rate
        lr_scheduler.step()


    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))    
    print(f"Training completed in {total_time_str}")

    print(f"Best MAE: {min_metrics[0]:.2f} at epoch {best_epoch}")
    
    with open(run_log_name, "a") as log_file:
        log_file.write(f"\nTraining completed.\n")
        log_file.write(f"Best MAE: {min_metrics[0]:.2f} achieved at epoch {best_epoch}\n")
        log_file.write(f"Total training time: {total_time_str}\n")


    # Close TensorBoard writer
    tb_logger.close()


if __name__ == '__main__':
    main(args)