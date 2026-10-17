# $ProjectRoot = "D:\coding\open_clip-main"
# & "D:\anaconda\envs\open_clip\python.exe" "D:\coding\open_clip-main\src\open_clip_train\main.py"
# $ENV:PYTHONPATH = "$ProjectRoot;$ENV:PYTHONPATH"
python "D:\coding\open_clip-main\src\open_clip_train\main.py"`
   --save-frequency 1 `
   --save-most-recent `
   --zeroshot-frequency 1 `
   --train-data 'D:\coding\open_clip-main\src\open_clip_train\raf_db_labels.csv' `
   --dataset-type csv `
   --csv_img_key image_path `
   --csv_caption_key caption `
   --lr "1e-5" `
   --beta1 0.9 `
   --beta2 0.95 `
   --warmup 3072 `
   --wd 0.2 `
   --batch-size 32 `
   --aug-cfg scale='(0.4, 1.0)' `
   --epochs 1 `
   --train-num-samples 131072 `
   --workers 1 `
   --model "ViT-B-16" `
   --pretrained 'D:\coding\open_clip-main\src\pretrained_models\ViT-B-16.pt' `
   --precision 'amp_bf16' `
   --ddp-static-graph `
   --local-loss `
   --gather-with-grad `
   --grad-checkpointing `
   --log-every-n-steps 256 `
   --seed 0 `
   --logs ./logs/ `
   --imagenet-val 'D:\coding\open_clip-main\src\open_clip_train\raf_db_labels.csv'
