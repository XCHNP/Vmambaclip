# train_finetune.ps1
# 单机单卡微调脚本（Windows PowerShell）

# ==== 环境配置 ====
# 1. 设置项目根目录和Python路径
# $ProjectRoot = "D:\coding\open_clip-main"
# & "D:\anaconda\envs\open_clip\python.exe" "D:\coding\open_clip-main\src\open_clip_train\main.py"
# $ENV:PYTHONPATH = "$ProjectRoot;$ENV:PYTHONPATH"
#
# # 2. 指定使用GPU设备
# $ENV:CUDA_VISIBLE_DEVICES = "0"

# ==== 训练参数配置 ====
# 模型配置

# ==== 启动训练命令 ====
python "D:\coding\open_clip-main\src\open_clip_train\main.py"`
    --train-data 'D:\coding\open_clip-main\src\open_clip_train\raf_db_labels.csv' `
    --val-data 'D:\coding\open_clip-main\src\open_clip_train\raf_db_labels.csv' `
    --dataset-type csv `
    --csv-img-key 'image_path' `
    --csv-caption-key 'caption' `
    --precision amp_bfloat16 `
    --batch-size 16 `
    --epochs 10 `
    --lr "1e-5" `
    --workers 1 `
    --model "ViT-B-16" `
    --pretrained 'D:\coding\open_clip-main\src\pretrained_models\ViT-B-16.pt' `
    --resume "latest" `
    --save-frequency 1 `
    --report-to tensorboard `
    --logs ./logs/ `
    --grad-checkpointing `
    --local-loss `
    --gather-with-grad `
    --seed 42 `

#     --validate-frequency 1 `
#     --imagenet-val "D:\datasets\RAF_DB\val_labels.csv"  # 可选验证集