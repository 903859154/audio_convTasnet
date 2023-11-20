import pathlib

from utils import dist_utils
from model import get_model
import dataset
import executor
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio
import torch.distributed as dist
import os
import time

_LG = dist_utils.getLogger(__name__)

def get_args():

    parser = ArgumentParser()
    #dataset options  #抄的是lightning—train 的cli_main里面的参数配置
    parser.add_argument("--dataset", default="librimix", type=str, choices=["wsj0mix", "librimix"])
    parser.add_argument("--root_dir",type=Path,
                        help="The path to the directory where the directory ``Libri2Mix`` or ``Libri3Mix`` is stored.",)
    parser.add_argument("--librimix-tr-split",default="train-360",choices=["train-360", "train-100"],
                        help="The training partition of librimix dataset. (default: ``train-360``)",)
    parser.add_argument("--librimix-task", default="sep_clean", type=str,
                        choices=["sep_clean", "sep_noisy", "enh_single", "enh_both"],
                        help="The task to perform (separation or enhancement, noisy or clean). (default: ``sep_clean``)",)
    #save options
    # parser.add_argument("--save-dir", default=Path("./exp"), required=True, type=pathlib.Path, help=("Directory where the checkpoints and logs are saved."
    #         "Though, only the worker 0 saves checkpoint data, "
    #         "all the worker processes must have access to the directory."))

    parser.add_argument("--exp",default=Path("./exp"), type=Path,
                        help="The directory to save checkpoints and logs.")
    parser.add_argument("--checkpoint", default=Path("./exp/checkpoint"), type=Path,
                        help="The directory to save checkpoints")
    #Training Options
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_speakers",default=2, type=int,
                        help="The number of speakers in the mixture. (default: 2)")
    parser.add_argument("--sample_rate",default=8000,type=int,
                        help="Sample rate of audio files in the given dataset. (default: 8000)",)
    parser.add_argument("--epochs",
                        metavar="NUM_EPOCHS",
                        default=120,
                        type=int,
                        help="The number of epochs to train. (default: 200)",)
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="Initial learning rate. (default: 1e-3)",)

    # 抄的是lightning—train 的cli_main里面的参数配置
    parser.add_argument("--num_gpu",default=1,type=int,
                        help="The number of GPUs for training. (default: 1)",)
    parser.add_argument("--num_node",default=1,type=int,
                        help="The number of nodes in the cluster for training. (default: 1)",)
    parser.add_argument("--num_workers",default=4,type=int,
                        help="The number of workers for dataloader. (default: 4)",
                        )
    parser.add_argument("--grad_clip", metavar="CLIP_VALUE", default=5.0,type=float,
                       help="Gradient clip value (l2 norm). (default: 5.0)",)
    parser.add_argument("--resume", metavar="./exp/checkpoint",
                       help="Previous checkpoint file from which the training is resumed.",)
    parser.add_argument("--debug", action="store_true", help="Enable debug log")

    args = parser.parse_args()
    return args

def main():
    #抄的是audio-conv-tasnet里的train里面的main
    args = get_args()
    args.exp.mkdir(parents=True, exist_ok=True)
    args.checkpoint.mkdir(parents=True, exist_ok=True)
    if "sox_io" in torchaudio.list_audio_backends():
        torchaudio.set_audio_backend("sox_io")

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.sample_rate != checkpoint["sample_rate"]:
            raise ValueError(
                "The provided sample rate ({args.sample_rate}) does not match "
                "the sample rate from the check point ({checkpoint['sample_rate']})."
            )
        if args.num_speakers != checkpoint["num_speakers"]:
            raise ValueError(
                "The provided #of speakers ({args.num_speakers}) does not match "
                "the #of speakers from the check point ({checkpoint['num_speakers']}.)"
            )
        start_epoch = checkpoint["epoch"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LG.info("Using: %s", device)
    model = get_model(num_sources=args.num_speakers)
    model.to(device)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'
    dist.init_process_group(backend='gloo', init_method='env://',rank=0, world_size=1) #bug: 调用torch.distributed下任何函数前，必须运行torch.distributed.init_process_group(backend='nccl')初始化。
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device] if torch.cuda.is_available() else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.resume:
        _LG.info("Loading parameters from the checkpoint...")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        dist_utils.synchronize_params(str(args.exp / "tmp.pt"), device, model, optimizer)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    train_loader, valid_loader, eval_loader = dataset.get_dataloader(
        args.dataset,
        args.root_dir,
        args.num_speakers,
        args.sample_rate,
        args.batch_size,
        args.num_workers,
        args.librimix_task,
        args.librimix_tr_split,
    )

    num_train_samples = len(train_loader)
    num_valid_samples = len(valid_loader)
    num_eval_samples = len(eval_loader)

    _LG.info_on_master("Datasets:")
    _LG.info_on_master(" - Train: %s", num_train_samples)
    _LG.info_on_master(" - Valid: %s", num_valid_samples)
    _LG.info_on_master(" - Eval: %s", num_eval_samples)

    executor1 = executor.Executor(
        model,
        optimizer,
        train_loader,
        valid_loader,
        eval_loader,
        args.grad_clip,
        device,
        debug=args.debug,
    )

    log_path = args.exp / "log.csv"
    dist_utils._write_header(log_path, args)
    dist_utils.write_csv_on_master(
        log_path,
        [
            "epoch",
            "learning_rate",
            "valid_si_snri",
            "valid_sdri",
            "eval_si_snri",
            "eval_sdri",
        ],
    )

    _LG.info_on_master("Running %s epochs", args.epochs)
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print("-------training start-------")
        start_time = time.time()
        _LG.info_on_master("=" * 70)
        _LG.info_on_master("Epoch: %s", epoch)
        _LG.info_on_master("Learning rate: %s", optimizer.param_groups[0]["lr"])
        _LG.info_on_master("=" * 70)
        print("Epoch: ",epoch, 'Learning rate: ', optimizer.param_groups[0]["lr"])

        #训练
        t0 = time.monotonic()
        executor1.train_one_epoch()
        train_sps = num_train_samples / (time.monotonic() - t0)

        _LG.info_on_master("-" * 70)
        train_min = (time.time() - start_time) / 60
        train_sec = (time.time() - start_time) % 60
        # print(train_min)
        # print(train_sec)
        print('Train Summary | End of Epoch {0} | Time：{1:.2f}min-{2:.2f}s | '.format(epoch, train_min,train_sec))
        print("============training over========")

        #验证
        print("-------validate start------")
        t0 = time.monotonic()
        valid_metric = executor1.validate()
        valid_sps = num_valid_samples / (time.monotonic() - t0)
        print("valid: ", valid_metric)
        val_min = (time.time() - start_time) / 60
        val_sec = (time.time() - start_time) % 60
        print('Val Summary | End of Epoch {0} | Time：{1:.2f}min-{2:.2f}s | '.format(epoch, val_min, val_sec))
        print("============validate over========")
        _LG.info_on_master("Valid: ", valid_metric)

        _LG.info_on_master("-" * 70)

        #评估
        print("-------validate start------")
        t0 = time.monotonic()
        eval_metric = executor1.evaluate()
        eval_sps = num_eval_samples / (time.monotonic() - t0)
        print("eval: %s: ", eval_metric)
        eval_min = (time.time() - start_time) / 60
        eval_sec = (time.time() - start_time) % 60
        print('Train Summary | End of Epoch {0} | Time：{1:.2f}min-{2:.2f}s | '.format(epoch, eval_min, eval_sec))
        _LG.info_on_master(" Eval: ", eval_metric)

        _LG.info_on_master("-" * 70)

        _LG.info_on_master("Train: Speed: %6.2f [samples/sec]", train_sps)
        _LG.info_on_master("Valid: Speed: %6.2f [samples/sec]", valid_sps)
        _LG.info_on_master(" Eval: Speed: %6.2f [samples/sec]", eval_sps)

        _LG.info_on_master("-" * 70)
        print('Epoch {} CV info valid_metric.si_snri, {}'.format(epoch, valid_metric.si_snri))
        print('Epoch {} CV info valid_metric.sdri, {}'.format(epoch, valid_metric.sdri))
        print('Epoch {} CV info eval_metric.si_snri, {}'.format(epoch, eval_metric.si_snri))
        print('Epoch {} CV info eval_metric.sdri, {}'.format(epoch, eval_metric.sdri))
        print('---------------------------')

        dist_utils.write_csv_on_master(
            log_path,
            [
                epoch,
                optimizer.param_groups[0]["lr"],
                valid_metric.si_snri,
                valid_metric.sdri,
                eval_metric.si_snri,
                eval_metric.sdri,
            ],
        )

        lr_scheduler.step(valid_metric.si_snri)

        save_path = args.checkpoint / f"epoch_{epoch}.pt"
        dist_utils.save_on_master(
            save_path,
            {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "num_speakers": args.num_speakers,
                "sample_rate": args.sample_rate,
                "epoch": epoch,
            },
        )
        print('Saved checkpoint model to ' + str(save_path) + 'Successfully')

if __name__ == '__main__':
    main()