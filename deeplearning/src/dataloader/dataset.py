import os
from torch.utils.data import Dataset
import cv2

#构建一个数据源类，对数据进行处理，可传入数据集文件位置及信息，数据增强的形式，进行数据集加载和数据增强
class MedicalDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
        train_file_dir="train.txt",
        val_file_dir="val.txt",
        test_file_dir="test.txt"
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform

        # 根据split类型加载不同数据列表
        if self.split == "train":
            list_file = train_file_dir
        elif self.split == "val":
            list_file = val_file_dir
        elif self.split == "test":
            list_file = test_file_dir
        else:
            raise ValueError("Invalid split type. Use 'train', 'val' or 'test'")

        # 从文件读取样本列表
        with open(os.path.join(self._base_dir, list_file), "r") as f:
            self.sample_list = [line.strip() for line in f] #去除首位空白字符，按行存入sample_list，一行一个样本
        print(f"Total {len(self.sample_list)} {self.split} samples")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        filename = f"{case}.png"  # 生成完整文件名

        # 加载图像
        image_path = os.path.join(self._base_dir, "images", filename)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 根据split决定是否加载标签
        label = None
        if self.split in ["train", "val"]:
            label_path = os.path.join(self._base_dir, "masks", "0", filename)
            assert os.path.exists(label_path), f"标签文件不存在：{label_path}"
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)[..., None] #以灰度模式读取掩码
            if label is None:
                raise FileNotFoundError(f"Label not found: {label_path}")

        # 数据增强/变换
        if self.split in ["train", "val"]:
            augmented = self.transform(image=image, mask=label)
            image = augmented["image"]
            label = augmented["mask"].astype("float32") / 255
            label = label.transpose(2, 0, 1)
        else:  # 测试模式只变换图像
            augmented = self.transform(image=image)
            image = augmented["image"]

        # 标准化图像
        image = image.astype("float32") / 255
        image = image.transpose(2, 0, 1)

        # 构建返回样本
        sample = {
            "image": image,
            "filename": filename
        }
        if self.split in ["train", "val"]:
            sample["label"] = label

        return sample