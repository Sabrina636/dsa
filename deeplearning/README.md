# Unet改进版（表现最优）

## 数据集
本数据集结构如下：
├── unet_last
| ├── data
|  |  ├── train_pre
|  |  |  ├── dsa_train.txt
|  |  |  ├── dsa_val.txt
|  |  |  ├── images
|  |  |  |  ├── #62 liu qiao hogn RCA RAO2 CRA39 Series9 _0000.png
|  |  |  |  ├── ...
|  |  |  ├── masks
|  |  |  |  ├── 0
|  |  |  |  |  ├── #62 liu qiao hogn RCA RAO2 CRA39 Series9 _0000.png
|  |  |  |  |  ├── ...

## 环境

 - GPU
 - Pytorch: 1.13.0 cuda 11.7
 - cudatoolkit: 11.7.1
 - scikit-learn: 1.0.2
 - albumentations: 1.2.0
 
## 需要的依赖
pip install completion ml-collections yacs cv2

## 运行步骤
在命令行运行以下代码

    python main.py --model U_Net --base_dir ./data/dsa_pre --train_file_dir dsa_train.txt --val_file_dir dsa_val.txt --base_lr 0.01 --epoch 500 --batch_size 32
或者，因为本模型训练较慢，建议挂在服务器后台运行

    nohup python main.py --model U_Net --base_dir ./data/dsa_pre --train_file_dir dsa_train.txt --val_file_dir dsa_val.txt --base_lr 0.01 --epoch 500 --batch_size 32 > train.log 2>&1 &
训练完毕后在测试集上进行预测,结果保存在validation_results

```
python infer.py --model U_Net --model_path ./checkpoint/U_Net_model.pth --base_dir ./data/test --split test --test_file_dir test_test.txt --img_size 256 --num_classes 1

```





