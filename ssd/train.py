import os
import argparse
import torch
import warnings
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from .utils.data.dataloader import create_dataloader
from .utils.misc import load_config, build_model, nms
from .utils.metrics import Mean, AveragePrecision
import mlflow
import numpy as np
import random
import mlflow.pytorch


class CheckpointManager(object):
    def __init__(self, logdir, model, optim, scaler, scheduler, best_score):
        self.epoch = 0
        self.logdir = logdir
        self.model = model
        self.optim = optim
        self.scaler = scaler
        self.scheduler = scheduler
        self.best_score = best_score

    def save(self, filename):
        data = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'best_score': self.best_score,
        }
        torch.save(data, os.path.join(self.logdir, filename))

    def restore(self, filename):
        data = torch.load(os.path.join(self.logdir, filename))
        self.model.load_state_dict(data['model_state_dict'])
        self.optim.load_state_dict(data['optim_state_dict'])
        self.scaler.load_state_dict(data['scaler_state_dict'])
        self.scheduler.load_state_dict(data['scheduler_state_dict'])
        self.epoch = data['epoch']
        self.best_score = data['best_score']

    def restore_lastest_checkpoint(self):
        if os.path.exists(os.path.join(self.logdir, 'last.pth')):
            self.restore('last.pth')
            print("Restore the last checkpoint.")


def get_lr(optim):
    for param_group in optim.param_groups:
        return param_group['lr']


def train_step(images, true_boxes, true_classes, model, optim, amp, scaler,
               metrics, device):
    images = images.to(device)
    true_boxes = [x.to(device) for x in true_boxes]
    true_classes = [x.to(device) for x in true_classes]

    optim.zero_grad()
    with autocast(enabled=amp):
        preds = model(images)
        loss = model.compute_loss(preds, true_boxes, true_classes)
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()

    loss = loss.item()
    metrics['loss'].update(loss, images.shape[0])


def test_step(images, true_boxes, true_classes, difficulties, model, amp,
              metrics, device):
    images = images.to(device)
    true_boxes = [x.to(device) for x in true_boxes]
    true_classes = [x.to(device) for x in true_classes]
    difficulties = [x.to(device) for x in difficulties]

    with autocast(enabled=amp):
        preds = model(images)
        loss = model.compute_loss(preds, true_boxes, true_classes)
    loss = loss.item()
    metrics['loss'].update(loss, images.shape[0])

    det_boxes, det_scores, det_classes = nms(*model.decode(preds))
    metrics['APs'].update(det_boxes, det_scores, det_classes,
                          true_boxes, true_classes, difficulties)


def main():
    seed_num = 42
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_name', default="Default", type=str)
    parser.add_argument('--cfg', default="ssd/configs/coco/ssd300.yaml", type=str, 
                        help="config file")
    parser.add_argument('--logdir', default="runs/coco_ssd300/exp0/", type=str,
                        help="log directory")
    parser.add_argument('--workers', type=int, default=8,
                        help="number of dataloader workers")
    parser.add_argument('--resume', action='store_true',
                        help="resume training")
    parser.add_argument('--no_amp', action='store_true',
                        help="disable automatic mix precision")
    parser.add_argument('--val_period', type=int, default=1,
                        help="number of epochs between successive validation")
    parser.add_argument('--dataset_path')
    parser.add_argument('--epochs')
    args = parser.parse_args()
    
    logdir = f"ssd/runs/coco_ssd300/{args.run_name}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    save_dir = len(os.listdir(logdir)) + 1
    args.logdir = f"{logdir}/{save_dir}"

    experiment_name ="SSD trains"
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=args.run_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.cfg)
    cfg.merge_from_list(['train_json', f'{args.dataset_path}/train.json'])
    cfg.merge_from_list(['val_json', f'{args.dataset_path}/val.json'])
    cfg.merge_from_list(['epochs', int(args.epochs)])
    enable_amp = (not args.no_amp)

    if os.path.exists(args.logdir) and (not args.resume):
        raise ValueError("Log directory %s already exists. Specify --resume "
                         "in command line if you want to resume the training."
                         % args.logdir)

    model = build_model(cfg)
    for key, value in cfg.items():
        mlflow.log_param(key, value)
        
    model.to(device)

    train_loader = create_dataloader(cfg.train_json,
                                     batch_size=cfg.batch_size,
                                     image_size=cfg.input_size,
                                     image_mean=cfg.image_mean,
                                     image_stddev=cfg.image_stddev,
                                     augment=True,
                                     shuffle=True,
                                     num_workers=args.workers,
                                     seed=seed_num)
    val_loader = create_dataloader(cfg.val_json,
                                   batch_size=cfg.batch_size,
                                   image_size=cfg.input_size,
                                   image_mean=cfg.image_mean,
                                   image_stddev=cfg.image_stddev,
                                   num_workers=args.workers,
                                   seed=seed_num)

    # Criteria
    optim = getattr(torch.optim, cfg.optim.pop('name'))(model.parameters(),
                                                        **cfg.optim)
    scaler = GradScaler(enabled=enable_amp)
    scheduler = getattr(torch.optim.lr_scheduler, cfg.scheduler.pop('name'))(
        optim,
        **cfg.scheduler
    )
    metrics = {
        'loss': Mean(),
        'APs': AveragePrecision(len(cfg.class_names), cfg.recall_steps)
    }

    # Checkpointing
    ckpt = CheckpointManager(args.logdir,
                             model=model,
                             optim=optim,
                             scaler=scaler,
                             scheduler=scheduler,
                             best_score=0.)
    ckpt.restore_lastest_checkpoint()

    # TensorBoard writers
    writers = {
        'train': SummaryWriter(os.path.join(args.logdir, 'train')),
        'val': SummaryWriter(os.path.join(args.logdir, 'val'))
    }

    # Kick off
    for epoch in range(ckpt.epoch + 1, cfg.epochs + 1):
        print("-" * 10)
        print("Epoch: %d/%d" % (epoch, cfg.epochs))

        # Train
        model.train()
        metrics['loss'].reset()
        if epoch == 1:
            warnings.filterwarnings(
                'ignore',
                ".*call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`.*"  # noqa: W605
            )
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=0.0001,
                total_iters=min(1000, len(train_loader))
            )
        pbar = tqdm(train_loader,
                    bar_format="{l_bar}{bar:20}{r_bar}",
                    desc="Training")
        for (images, true_boxes, true_classes, _) in pbar:
            train_step(images,
                       true_boxes,
                       true_classes,
                       model=model,
                       optim=optim,
                       amp=enable_amp,
                       scaler=scaler,
                       metrics=metrics,
                       device=device)
            loss = metrics['loss'].result
            lr = get_lr(optim)
            pbar.set_postfix(loss='%.5f' % metrics['loss'].result, lr=lr)

            if epoch == 1:
                warmup_scheduler.step()
        writers['train'].add_scalar('Loss', loss, epoch)
        writers['train'].add_scalar('Learning rate', get_lr(optim), epoch)
        mlflow.log_metric("train_loss", loss, step=epoch)
        mlflow.log_metric("lr", get_lr(optim), step=epoch)
        scheduler.step()

        # Validation
        if epoch % args.val_period == 0:
            model.eval()
            metrics['loss'].reset()
            metrics['APs'].reset()
            pbar = tqdm(val_loader,
                        bar_format="{l_bar}{bar:20}{r_bar}",
                        desc="Validation")
            with torch.no_grad():
                for (images, true_boxes, true_classes, difficulties) in pbar:
                    test_step(images,
                              true_boxes,
                              true_classes,
                              difficulties,
                              model=model,
                              amp=enable_amp,
                              metrics=metrics,
                              device=device)
                    pbar.set_postfix(loss='%.5f' % metrics['loss'].result)
            APs = metrics['APs'].result
            mAP50 = APs[:, 0].mean()
            mAP = APs.mean()
            if mAP >= ckpt.best_score:
                ckpt.best_score = mAP
                ckpt.save('best.pth')
                mlflow.pytorch.log_model(model, "best_model")
            print("mAP@[0.5]: %.3f" % mAP50)
            print("mAP@[0.5:0.95]: %.3f (best: %.3f)" % (mAP, ckpt.best_score))
            writers['val'].add_scalar('Loss', metrics['loss'].result, epoch)
            writers['val'].add_scalar('mAP@[0.5]', mAP50, epoch)
            writers['val'].add_scalar('mAP@[0.5:0.95]', mAP, epoch)
            mlflow.log_metric("val_loss", metrics['loss'].result, step=epoch)
            mlflow.log_metric("val_mAP_0.5", mAP50, step=epoch)
            mlflow.log_metric("mAP_0.5-0.95_", mAP, step=epoch)


        ckpt.epoch += 1
        ckpt.save('last.pth')

    writers['train'].close()
    writers['val'].close()
    mlflow.end_run()


if __name__ == '__main__':
    main()
