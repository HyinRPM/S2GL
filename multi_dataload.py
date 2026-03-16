import torch
from torch.utils.data import Dataset, DataLoader


# 假设您有一个包含多模态数据和标签的数据集类 MultimodalDataset
class MultimodalDataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = data1  # 第一种模态的数据
        self.data2 = data2  # 第二种模态的数据
        self.labels = labels  # 数据对应的标签

    def __len__(self):
        return len(self.data1)  # 假设两种模态的数据长度相同

    def __getitem__(self, index):
        return self.data1[index], self.data2[index], self.labels[index]  # 返回对应索引的两种模态数据和标签


if __name__ == "__main__":
    # 只有文件作为脚本直接运行时，才会执行如下代码
    # 假设 data1 是第一种模态的数据，data2 是第二种模态的数据，labels 是数据对应的标签
    data1 = [1, 2, 3, 4, 5]
    data2 = ['a', 'b', 'c', 'd', 'e']
    labels = [0, 1, 0, 1, 1]

    multimodal_dataset = MultimodalDataset(data1, data2, labels)
    data_loader = DataLoader(dataset=multimodal_dataset, batch_size=3, shuffle=True)

    for batch in data_loader:
        data1_batch, data2_batch, labels_batch = batch
        print("Data1 batch:", data1_batch)
        print("Data2 batch:", data2_batch)
        print("Labels batch:", labels_batch)
        print()
