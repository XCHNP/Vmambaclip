#torchrun --nproc_per_node 8 -m open_clip_train.main \
##python -m open_clip_train.main \
#    --save-frequency 1 \
#    --save-most-recent \
#    --zeroshot-frequency 1 \
#    --train-data '\src\open_clip_train\raf_db_labels.csv' \
#    --dataset-type webdataset \
#    --lr "2.56e-5" \
#    --beta1 0.9 \
#    --beta2 0.95 \
#    --warmup 3072 \
#    --wd 0.2 \
#    --batch-size 1024 \
#    --aug-cfg scale='(0.4, 1.0)' \
#    --epochs 1 \
#    --train-num-samples 131072000 \
#    --workers 6 \
#    --model ViT-B-16-CL16 \
#    --pretrained '/path/to/ckpt' \
#    --precision 'amp_bf16' \
#    --ddp-static-graph \
#    --local-loss \
#    --gather-with-grad \
#    --grad-checkpointing \
#    --log-every-n-steps 256 \
#    --seed 0 \
#    --logs ./logs/ \
#    --imagenet-val '/path/to/imagenet/val'


python -m open_clip_train.main `
    --save-frequency 1 `
    --save-most-recent `
    --zeroshot-frequency 1 `
    --train-data "D:\coding\open_clip_train\raf_db_labels.csv" `
    --dataset-type csv `
    --lr "2.56e-5" `
    --beta1 0.9 `
    --beta2 0.95 `
    --warmup 3072 `
    --wd 0.2 `
    --batch-size 32 `  # 根据单GPU显存调整
    --aug-cfg scale='(0.4, 1.0)' `
    --epochs 1 `
    --train-num-samples 131072 `
    --workers 4 `  # Windows建议减少workers数量
    --model ViT-B-16-CL16 `
    --pretrained "D:\pretrained_models\your_checkpoint.pt" `
    --precision "amp_bf16" `
    --local-loss `
    --gather-with-grad `
    --grad-checkpointing `
    --log-every-n-steps 256 `
    --seed 0 `
    --logs "./logs/" `
    --imagenet-val "D:\datasets\imagenet\val"


# $env:PYTHONPATH = "D:\coding\open_clip-main\src\open_clip_train;$env:PYTHONPATH"
# python -m open_clip_train.main `
#     --save-frequency 1 `
#     --save-most-recent `
#     --zeroshot-frequency 1 `
#     --train-data "D:\\coding\\open_clip_train\\raf_db_labels.csv" `
#     --dataset-type-csv `
#     --lr "2.56e-5" `
#     --beta1 0.9 `
#     --beta2 0.95 `
#     --warmup 3072 `
#     --wd 0.2 `
#     --batch-size 32 `  # 根据单GPU显存调整
#     --aug-cfg scale='(0.4, 1.0)' `
#     --epochs 1 `
#     --train-num-samples 131072 `
#     --workers 4 `  # Windows建议减少workers数量
#     --model ViT-B-16-CL16 `
#     --pretrained "D:\\coding\\open_clip-main\\src\\pretrained_models\\ViT-B-16.pt" `
#     --precision "amp_bf16" `
#     --local-loss `
#     --gather-with-grad `
#     --grad-checkpointing `
#     --log-every-n-steps 256 `
#     --seed 0 `
#     --logs "./logs/" `
#     --imagenet-val "D:\\coding\\open_clip_train\\raf_db_labels.csv"


# torchrun --nproc_per_node 8 -m open_clip_train.main \
#     python -m open_clip_train.main --save-frequency 1 --save-most-recent --zeroshot-frequency 1 --train-data "D:\\coding\\open_clip_train\\raf_db_labels.csv" --dataset-type csv --lr "2.56e-5"--beta1 0.9 --beta2 0.95 --warmup 3072 --wd 0.2 --batch-size 32 --aug-cfg scale='(0.4, 1.0)' --epochs 1 --train-num-samples 131072 --workers 4  --model ViT-B-16-CL16 --pretrained "D:\\coding\\open_clip-main\\src\\pretrained_models\\ViT-B-16.pt" --precision "amp_bf16" --local-loss --gather-with-grad --grad-checkpointing --log-every-n-steps 256 --seed 0 --logs "./logs/" --imagenet-val "D:\\coding\\open_clip_train\\raf_db_labels.csv"