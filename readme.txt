1. 准备数据
    1）slim/datasets/download_and_convert_products.py
    修改参数：
        _NUM_VALIDATION 验证集的图片数目
        _NUM_SHARDS 生成tf-records的数目（可以自行确定）
        201-202行是转化训练集，203-204是转化验证集
    2）slim/datasets/products.py
    修改参数：
        SPLITS_TO_SIZES = {'train': 422, 'validation': 0} #对应为训练集和验证集的图片数目

    3）假设数据路径为/home/1.7-code, 1.7-code下目录结构为：
        product_images/          存放图片，每个子文件夹为一类
        tf-records/              存放生成的tf-records

    4)运行slim/download_and_convert_data.py, 注意参数dataset_dir为你自己的数据路径
        python download_and_convert_data.py --dataset_dir="/home/1.7-code/"

2. 训练
    运行slim/train_image_classifier.py, 注意参数dataset_dir为你自己的数据路径,train_dir为训练模型存储的路径
        python train_image_classifier.py --dataset_dir="/home/1.7-code" --train_dir="/home/1.7-code/training/" --train_image_size="299" (optional)

3. 测试
    运行slim/eval_image_classifier.py, 注意参数dataset_dir为你自己的数据路径,checkpoint_path为训练模型存储的路径
        python eval_image_classifier.py --dataset_dir="/home/1.7-code" --checkpoint_path="/home/1.7-code/training/"  --eval_dir="/home/1.7-code/eval/"
