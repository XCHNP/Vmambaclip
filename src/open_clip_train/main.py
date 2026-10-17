
import copy
import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import optim

try:
    import wandb # Weights & Biases 是一个用于实验跟踪和可视化的工具，可帮助用户记录和分析模型训练过程中的各种指标
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard # PyTorch 集成的 TensorBoard 工具，用于可视化模型训练过程中的数据，如损失曲线、准确率等
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd  # Horovod 是一个用于分布式训练的框架，可加速深度学习模型的训练过程
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown
from open_clip_train.train import train_one_epoch, evaluate
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync


LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    # 1、参数解析与环境设置
    args = parse_args(args)  # parse_args：解析命令行参数，获取训练或评估所需的配置

    if torch.cuda.is_available():  # CUDA 相关设置
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)  # 初始化分布式训练环境，确定当前进程使用的设备

    # 2、实验名称与日志设置
    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    # 3、远程同步与检查点恢复
    # 检查点恢复
    resume_latest = args.resume == 'latest' # 通过比较args.resume是否等于latest，判断是否要恢复最新的检查点。resume_latest(bool)
    log_base_path = os.path.join(args.logs, args.name) # 得到本次实验日志文件存储的基础路径，存储在 log_base_path 变量中
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True) # os.makedirs用于递归创建目录:log_base_path 是前面确定的日志文件基础路径。exist_ok=True 表示如果目录已经存在，不会抛出异常，而是直接跳过创建操作
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename) # 组合成完整的日志文件路径并赋值
        if os.path.exists(args.log_path) and not resume_latest: # 如果日志文件已经存在，并且 resume_latest 为 False（即不恢复最新检查点）
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )# 打印错误信息
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from
    # 代码复制
    if args.copy_codebase: # 如果 copy_codebase 参数为真，则复制代码库到日志目录
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()



    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    # 分布式训练模式提示
    if args.horovod: # Horovod 模式
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed: # 分布式模式
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else: # 单进程模式
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    # distill_model 通常指定要使用的蒸馏模型,distill_pretrained 可能指定该蒸馏模型的预训练权重
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.

        # args.accum_freq 通常表示梯度累积的频率，梯度累积是一种在内存有限的情况下模拟大批次训练的技术
        assert args.accum_freq == 1 # 当前代码还不支持在进行模型蒸馏时使用梯度累积技术
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()  # 当前代码还不支持使用 CoCa 模型进行蒸馏
     # 图像尺寸参数处理
    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1: # 将用户可能输入的单元素列表或元组形式的图像尺寸转换为单个整数
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    # 设置随机种子：确保模型的初始化参数、数据的随机打乱等操作都是一致的
    random_seed(args.seed, 0)

    # 4、模型创建与配置
    model_kwargs = {}
    if args.siglip: # 根据 siglip 参数添加额外模型参数
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10

    print(f'will create model',args.model)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device, # 指定模型运行的设备，如 'cuda' 或 'cpu'
        jit=args.torchscript, # 指定是否使用 TorchScript 对模型进行编译
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image, # 指定是否使用预训练的图像编码器
        output_dict=True,
        cache_dir=args.cache_dir, # 指定缓存目录，用于存储预训练模型权重等文件
        **model_kwargs, # 将 model_kwargs 字典中的键值对作为关键字参数传递给 create_model_and_transforms 函数
    )
    if args.distill: # 创建一个用于知识蒸馏的教师模型
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        # 当前代码假设用于知识蒸馏的模型（即被蒸馏的教师模型）和目标模型（学生模型）使用相同的分词器（对于处理文本数据时）和数据转换（如图像预处理）。在实际应用中，这可能并不总是成立，需要后续进行改进
        dist_model, _, _ = create_model_and_transforms( # dist_model, _, _ 这种形式是 Python 的解包操作。dist_model 接收创建好的蒸馏模型，而后面两个下划线 _ 表示忽略返回的训练数据预处理函数和验证数据预处理函数
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )
    if args.use_bnb_linear is not None: # 检查是否使用 bitsandbytes 线性层替换  使用 bitsandbytes 库中的特定线性层来替换模型中的普通线性层
        print('=> using a layer from bitsandbytes.\n' # 会打印提示信息告知用户正在使用 bitsandbytes 库中的层
              '   this is an experimental feature which requires two extra pip installs\n' # 这是一个实验性的特性，需要额外安装 bitsandbytes 和 triton 库
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0') # 提醒用户要使用 triton 2.0.0 版本
        import bitsandbytes as bnb
        from ..open_clip.utils import replace_linear # replace_linear 函数，作用是将模型中的普通线性层替换为指定的线性层
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear) # getattr 函数用于从 bnb.nn.triton_based_modules 中获取 args.use_bnb_linear 所指定的类
        # 调用 replace_linear 函数，将模型 model 中的所有普通线性层（nn.Linear）替换为 linear_replacement_cls 所指定的线性层
        replace_linear(model, linear_replacement_cls)
        model = model.to(device) # 将替换后的模型移动到指定的设备（如 GPU 或 CPU）上
    # 设置随机种子
    random_seed(args.seed, args.rank) # args.rank 通常用于分布式训练中标识不同的进程

    # 模型追踪
    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    # 锁定图像编码器
    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)

    # 锁定文本编码器
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    # 启用梯度检查点
    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args): # 判断当前进程是否为主进程
        # 为了避免多个进程重复记录信息，只让主进程来执行记录操作
        # logging.info(f"{str(model)}") # str(model) 会输出模型的详细结构，包括模型包含的各个层及其参数设置等
        logging.info("main 310 !!!! Params:") # 提示接下来记录的是命令行参数的信息
        # 最终生成的文件路径为 params.txt，用于存储命令行参数信息
        params_file = os.path.join(args.logs, args.name, "params.txt") # os.path.join 函数拼接文件路径
        # 写入参数信息到文件
        with open(params_file, "w") as f:
            for name in sorted(vars(args)): # vars(args) 会返回 args 对象的所有属性和属性值组成的字典，sort对key排序，
                # 通过遍历排序后的参数名，依次处理每个参数
                val = getattr(args, name) # 使用 getattr 函数获取 args 对象中指定名称的属性值
                logging.info(f"  {name}: {val}") # 使用 logging 模块记录每个参数的名称和对应的值，方便在控制台查看
                f.write(f"{name}: {val}\n") # 将每个参数的名称和对应的值写入到 params_file 文件中，每行记录一个参数信息，格式为 参数名: 参数值

    # 处理在分布式训练环境下对模型进行的一系列设置
    if args.distributed and not args.horovod: # args.distributed(bool),为 True 时,表示要启用分布式训练；horovod 是另一种用于分布式训练的框架
    # 这里 not args.horovod 表示不使用 horovod 进行分布式训练，而是使用 PyTorch 自带的 DistributedDataParallel 方法
        # 同步批量归一化
        if args.use_bn_sync: #true时表示要使用同步批量归一化
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # 将模型中的普通批量归一化层（BatchNorm）转换为同步批量归一化层（SyncBatchNorm）
        # 设置 DDP 参数
        ddp_args = {}
        if args.ddp_static_graph:# true表示要使用静态图优化
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True # 如果存在，将键值对添加到ddp_args 字典中
        # torch.nn.parallel.DistributedDataParallel：这是 PyTorch 提供的用于分布式训练的类，它会将模型包装成一个支持分布式训练的对象
        # model代表学生模型
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

        # 对蒸馏模型进行同样的处理（如果启用蒸馏）
        if args.distill:
            # dist_model代表教师模型
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)


    # 根据用户的配置创建优化器（optimizer）和梯度缩放器（scaler）
    # create optimizer and scaler
    optimizer = None # 优化器用于在训练时更新模型的参数
    scaler = None # 梯度缩放器在混合精度训练中发挥作用，用于防止梯度下溢

    # 检查训练条件和模型追踪
    if args.train_data or args.dataset_type == "synthetic": # 检查是否提供了训练数据（args.train_data 有值）或者数据集类型为合成数据
        # 使用 assert 语句来确保没有对模型进行追踪。因为经过追踪的模型是静态图，不适合用于训练，若 args.trace 为 True，会抛出错误信息
        assert not args.trace, 'Cannot train with traced model'

        # 选择优化器类型并创建优化器
        opt = getattr(args, 'opt', 'adamw').lower() # 从 args 中获取优化器的名称，若未指定则默认为 'adamw'，并将其转换为小写
        if opt.startswith('timm/'): # 检查优化器名称是否以 'timm/' 开头
            from timm.optim import create_optimizer_v2 # 导入 timm 库中的 create_optimizer_v2 函数
            timm_opt = opt.split('timm/')[-1] # 提取 timm 优化器的具体名称
            opt_kwargs = {} # 存储传递给 create_optimizer_v2 的额外参数
            # args.beta1 和 args.beta2 通常是 Adam 及其变体（如 AdamW）优化器中的超参数
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2) # 将 (args.beta1, args.beta2) 作为 betas 参数添加到 opt_kwargs 中
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum # 将其作为 momentum 参数添加到 opt_kwargs 中
            optimizer = create_optimizer_v2( # 调用 create_optimizer_v2 函数创建优化器，传入模型、优化器名称、学习率、权重衰减、数值稳定性参数以及额外参数
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else: # 使用 PyTorch 原生的 AdamW 优化器
            # If some params are not passed, we use the default values based on model name.
            # 定义 exclude 和 include 两个匿名函数，用于筛选模型的参数
            # exclude 函数筛选出维度小于 2、名称包含 "bn"（批量归一化）、"ln"（层归一化）、"bias"（偏置）或 'logit_scale' 的参数
            exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
            # include 函数则筛选出不满足 exclude 条件的参数
            include = lambda n, p: not exclude(n, p)

            # 获取模型的所有命名参数
            named_parameters = list(model.named_parameters())
            # 分别存储满足 exclude 和 include 条件且需要梯度更新的参数
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

            if opt == 'adamw': # 若优化器名称为 'adamw'，则使用 optim.AdamW 创建优化器，并为不同类型的参数设置不同的权重衰减
                optimizer = optim.AdamW(
                    [
                        {"params": gain_or_bias_params, "weight_decay": 0.},
                        {"params": rest_params, "weight_decay": args.wd},
                    ],
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
            else: # 若优化器名称不是 'adamw'，则抛出错误
                assert False, f'Unknown optimizer {opt}'

        # 记录优化器信息
        if is_master(args): # 若当前进程是主进程，则记录创建的优化器的信息，包括优化器类型和默认参数
            if is_master(args):
                defaults = copy.deepcopy(optimizer.defaults)
                defaults['weight_decay'] = args.wd
                defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
                logging.info(
                    f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
                )

        # 使用 Horovod 进行分布式优化
        if args.horovod:
            # 将普通优化器包装成 Horovod 的分布式优化器
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            # 将根进程（root_rank=0）的模型参数广播到所有进程，确保所有进程的模型参数初始值一致。
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            # 将根进程的优化器状态广播到所有进程，保证所有进程的优化器状态一致
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        # 创建梯度缩放器
        scaler = None
        if args.precision == "amp": # 若 args.precision 为 "amp"，表示使用自动混合精度训练
            try: # 使用 torch.amp.GradScaler(device=device) 创建梯度缩放器
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e: # 若异常，使用 torch.cuda.amp.GradScaler() 创建
                scaler = torch.cuda.amp.GradScaler()

    # 根据用户的配置，从指定的检查点（checkpoint）文件中恢复模型的训练状态
    # 包括模型参数、优化器状态、训练轮次（epoch）等信息，以便继续之前的训练过程或者进行微调、评估等操作
    # optionally resume from a checkpoint
    start_epoch = 0 # 表示从第 0 个轮次开始训练。如果后续从检查点恢复训练，这个值会被更新为检查点中的轮次信息
    # 检查是否需要从检查点恢复
    if args.resume is not None: # args.resume 是一个命令行参数，用于指定检查点文件的路径
        # pt_load 是一个自定义函数，用于加载检查点文件
        # checkpoint 是一个字典，包含了检查点文件中的所有信息
        checkpoint = pt_load(args.resume, map_location='cpu') # map_location='cpu' 表示将检查点文件中的数据加载到 CPU 内存中，这样可以避免在 GPU 显存不足的情况下出现错误
        # 检查检查点文件的类型
        if 'epoch' in checkpoint: # 包含轮次和优化器状态的训练检查点
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"] # 从检查点字典中获取之前训练到的轮次，并将其赋值给 start_epoch，以便后续从该轮次继续训练
            sd = checkpoint["state_dict"] # 获取检查点中的模型状态字典，该字典包含了模型的所有参数
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):# 如果不是分布式训练（not args.distributed），并且状态字典中的键以 'module' 开头，说明该状态字典是在分布式训练时保存的
                # 去除 'module.' 前缀，以保证可以正确加载到非分布式模型中
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd) # 将处理后的状态字典加载到模型中，恢复模型的参数
            if optimizer is not None: # 如果优化器已经创建
                # 加载检查点中的优化器状态，包括学习率、动量等参数，以便继续之前的优化过程
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint: # 果使用了梯度缩放器，并且检查点中包含 'scaler' 信息
                # 加载梯度缩放器的状态，确保梯度缩放过程的连续性
                scaler.load_state_dict(checkpoint['scaler'])
            # 使用 logging.info 记录从指定检查点恢复训练的信息，包括检查点文件的路径和恢复的轮次
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else: # 仅包含模型参数的检查点（不含'epoch'信息）
        # 说明该检查点只包含模型的参数，通常用于微调（fine-tune）或者评估（evaluation）
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint) # 加载模型参数
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})") # 记录加载检查点的信息

    # initialize datasets
    # 初始化数据集，包括获取分词器和加载具体的数据
      # 获取分词器，赋值给 tokenizer 变量
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir) # cache_dir=args.cache_dir：指定分词器的缓存目录
      # 加载数据集
    data = get_data( # get_data：一个自定义函数，用于根据用户提供的参数加载训练集和验证集数据
        args,
        (preprocess_train, preprocess_val), # 这是一个元组，包含了训练数据和验证数据的预处理函数
        epoch=start_epoch, # 在分布式训练中，不同的轮次可能需要对数据进行不同的划分
        tokenizer=tokenizer, # 将之前获取的分词器传递给 get_data 函数，以便在处理文本数据时进行分词操作
    ) # get_data 函数返回一个包含训练集和验证集数据的对象，将其赋值给 data 变量
    assert len(data), 'At least one train or eval dataset must be specified.'
    # 这里检查 data 对象的长度是否大于 0，如果为 0 说明没有成功加载到任何训练集或验证集数据，此时会抛出异常并显示错误信息 'At least one train or eval dataset must be specified.'，提示用户至少需要指定一个训练集或验证集


    # create scheduler if train
    # 根据训练数据和用户指定的学习率调度策略，为优化器创建一个学习率调度器（Scheduler）。
    # 学习率调度器用于在训练过程中动态调整学习率，以提高模型的训练效果。
    scheduler = None # 初始化学习率调度器
    if 'train' in data and optimizer is not None: # 存在训练数据集(data 字典中包含 'train' 键)且优化器已创建：才创建学习率调度器
        # data["train"].dataloader.num_batches：表示训练数据加载器中的批次数量，即一个 epoch 中需要进行的迭代次数
        # args.accum_freq：是梯度累积频率。梯度累积允许在多个批次上累积梯度，然后再进行一次参数更新，这样可以模拟更大的批次大小
        # data["train"].dataloader.num_batches // args.accum_freq 计算出考虑梯度累积后，一个 epoch 中实际的参数更新次数
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs # 计算总训练步数：将一个 epoch 中实际的参数更新次数乘以总轮数

        # 根据学习率调度策略创建调度器
        if args.lr_scheduler == "cosine": # 用户通过命令行参数指定的学习率调度策略，"cosine"表示使用余弦退火调度策略
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps) # 是一个自定义函数，用于创建余弦退火学习率调度器
        elif args.lr_scheduler == "const": # 使用常数学习率调度策略
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":# 带冷却期的常数学习率调度器
            # 确保用户指定了冷却期的轮数（args.epochs_cooldown）
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule." # 如果没有指定，会抛出断言错误并给出提示信息
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown # 计算冷却期的总步数，即冷却期的轮数乘以考虑梯度累积后一个 epoch 中实际的参数更新次数
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error( # 记录错误日志并退出程序，提示用户可用的调度策略选项
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # 确定当前进程是否应该保存日志和检查点，以及初始化日志记录工具（如 TensorBoard 和 Weights & Biases）
    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args) # args.logs：这是一个命令行参数，用于指定日志保存的路径。如果所有条件都满足，则将 args.save_logs 设置为 True，表示当前进程需要保存日志
    # 初始化 TensorBoard 日志写入器
    writer = None
    if args.save_logs and args.tensorboard:# 检查是否需要保存日志且用户指定使用 TensorBoard 记录日志
        assert tensorboard is not None, "Please install tensorboard."  # 确保 tensorboard 模块已经成功导入
        writer = tensorboard.SummaryWriter(args.tensorboard_path) # 如果条件满足，则创建一个 SummaryWriter 对象，用于将训练过程中的各种指标（如损失值、准确率等）写入到指定的 args.tensorboard_path 路径下，以便后续使用 TensorBoard 可视化这些指标

    # 初始化 Weights & Biases（WandB）
    # wandb（Weights & Biases）是一个用于机器学习实验跟踪和可视化的工具和平台
    if args.wandb and is_master(args): # 检查用户是否指定使用 Weights & Biases 记录实验信息，并且当前进程为主进程。
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.') # 记录调试信息，表示开始初始化 Weights & Biases
        args.train_sz = data["train"].dataloader.num_samples # 获取训练数据集的样本数量，并将其赋值给 args.train_sz
        if args.val_data is not None: # 如果存在验证数据集
            args.val_sz = data["val"].dataloader.num_samples # 获取验证数据集的样本数量，并将其赋值给 args.val_sz
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name, # 指定实验所属的项目名称
            name=args.name, # 指定实验的名称
            id=args.name, # 指定实验的唯一标识符
            notes=args.wandb_notes, # 添加关于实验的备注信息
            tags=[], # 为实验添加标签，方便后续筛选和分类
            resume='auto' if args.resume == "latest" else None, # 如果 args.resume 为 "latest"，则自动恢复最近一次的实验；否则不进行恢复
            config=vars(args), # 将命令行参数以字典形式传递给 Weights & Biases，方便记录实验配置
        )
        if args.debug:# 如果用户指定了调试模式
            wandb.watch(model, log='all') # wandb.watch 函数监控模型的参数和梯度，并记录所有信息
        wandb.save(params_file) # 将保存命令行参数的文件（params_file）上传到 Weights & Biases，方便后续查看实验配置
        logging.debug('Finished loading wandb.') # 记录调试信息，表示 Weights & Biases 初始化完成


    # 实现了模型的编译、评估、训练以及检查点保存等功能，同时支持使用 PyTorch 2.0 的编译特性和一些额外的优化设置

    # 在 PyTorch 2.0 中，编译后的模型的 state_dict() 键会添加 '_orig_mod.' 前缀 Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile: # 检查是否需要对模型进行编译
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:# 同时启用了梯度检查点（args.grad_checkpointing 为 True）和分布式训练（args.distributed 为 True）
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False # 禁用 DDP（DistributedDataParallel）的动态优化器

        model = torch.compile(original_model) # 使用 torch.compile() 函数对原始模型进行编译，以提高模型的运行效率

    if 'train' not in data: # 检查数据中是否包含训练集。如果不包含训练集，则进入仅评估模式
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None: # 如果使用了 bnb_linear（可能是 8 位量化的线性层）
            from ..open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model) # 调用 convert_int8_model_to_inference_mode 函数将模型转换为推理模式
        # Evaluate.
        # 调用 evaluate 函数对模型进行评估，并传入模型、数据、起始轮次、参数、TensorBoard 写入器和分词器等信息
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

# 模型的训练循环、验证、检查点保存以及最终的远程同步等
    loss = create_loss(args) # 创建损失函数

    # 训练循环
    for epoch in range(start_epoch, args.epochs): # 从起始轮次 start_epoch 开始，到指定的总轮次 args.epochs 结束，进行迭代训练
        if is_master(args):
            logging.info(f'Start epoch {epoch}') # 如果是主进程，则记录当前轮次开始的日志信息

        # 调用 train_one_epoch 函数进行一轮训练，传入模型、数据、损失函数、当前轮次、优化器、梯度缩放器、学习率调度器、蒸馏模型、参数和 TensorBoard 写入器等信息
        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1 # 完成当前轮次训练后，更新已完成的轮次

        # 验证模型
        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')): # 检查数据中是否包含验证集（'val'）、ImageNet 验证集（'imagenet-val'）或 ImageNet V2 验证集（'imagenet-v2'）
            # 调用 evaluate 函数对模型进行评估，传入模型、数据、已完成的轮次、参数、TensorBoard 写入器和分词器等信息
            evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)

        # Saving checkpoints.
        # 保存检查点
        if args.save_logs: # 查是否需要保存日志和检查点
            checkpoint_dict = { # 创建一个字典，包含当前轮次、实验名称、原始模型的状态字典和优化器的状态字典
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None: # 如果使用了梯度缩放器
                checkpoint_dict["scaler"] = scaler.state_dict() # 包含其状态字典

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ): # 如果当前轮次是最后一轮，或者满足保存频率的条件（args.save_frequency 大于 0 且当前轮次能被保存频率整除）
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                ) # 将检查点字典保存到指定路径
            if args.delete_previous_checkpoint: # 如果需要删除上一轮的检查点
                # previous_checkpoint 变量存储的是上一轮检查点文件的完整路径
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):# 检查上一轮检查点文件的路径 previous_checkpoint 是否存在
                    os.remove(previous_checkpoint) # 存在则删除上一轮的检查点文件

            if args.save_most_recent: # 如果需要保存最新的检查点
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt") # 将检查点字典保存到临时文件 tmp.pt
                # 将临时文件重命名为最新检查点的文件名，避免保存失败导致文件损坏
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    # 结束 Weights & Biases 会话
    if args.wandb and is_master(args):# 如果使用了 Weights & Biases 记录实验信息，并且当前进程是主进程
        wandb.finish() # 调用 wandb.finish() 函数结束会话

    # 最终远程同步
    # run a final sync.
    if remote_sync_process is not None: # 检查是否存在远程同步进程
        logging.info('Final remote sync.')
        remote_sync_process.terminate() # 终止远程同步进程
        # 调用 remote_sync 函数进行远程同步，传入本地日志路径、远程同步路径和远程同步协议等信息
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        # 根据同步结果记录相应的日志信息
        if result: # 如果同步成功
            logging.info('Final remote sync successful.')
        else: # 同步失败
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    print(sys.argv[1:])
    main(sys.argv[1:])  # 将命令行中除脚本名称之外的所有参数传递给 main 函数
    # sys.argv 是一个列表，它包含了命令行参数
    # sys.argv[1:] 是从第一个命令行参数开始的所有参数
