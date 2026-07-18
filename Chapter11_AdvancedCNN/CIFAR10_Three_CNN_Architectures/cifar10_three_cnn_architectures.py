# 对比CNN中 普通CNN,Inception 和 Resnet 三种对cifar10的效果

# 首先导入需要的包

# 随机数工具，用于设置随机种子后，多次运行代码时，随机结果更容易保持一致
# 在我们的实验中它主要用于保证：
# 三个模型使用相同的随机条件
# 实验结果更容易重复
import random

# 用于记录时间 比如start_time,end_time
# 用来记录代码执行了多少秒
# 在比较模型时不能只看准确率，也要看训练成本
import time

# 用来复制python对象
import copy

# Numpy是python中常用的数值计算库
# 在此实验中主要用它设置随机种子np.random.seed(42)
import numpy as np
import torch

# 常用的绘图库
import matplotlib.pyplot as plt

# nn是Pytorch中专门用来搭建神经网络的模块
from torch import nn
from torch.nn import BatchNorm2d

# subset用来划分训练集和数据集
# 为什么不用random_split,比如创建一个完整的数据集
# full_dataset = datasets.CIFAR10(root="./data",train=True,download=True,transform=train_transform)
#这里的train_dataset和valid_dataset都来自于同一个full_dataset
# 也就是说验证集也会运行transform，这不合适
# Subset 是底层工具，random_split 是更方便的自动切割工具

from torch.utils.data import DataLoader,Subset

# torchvision 是 PyTorch 专门处理计算机视觉任务的库
# transform主要用于对图片进行预处理和数据增强
from torchvision import datasets,transforms

# 固定随机种子
# 设置一个固定的随机种子数值
# 42本身没有特殊含义，换成1，100，1024也可以
seed = 42

# 固定python自带的random模块产生的随机效果
random.seed(seed)

# 固定Numpy产生的随机效果
np.random.seed(seed)

# 固定Pytorch在CPU上产生的随机效果
# 神经网络创建时，卷积核权重通常是随机初始化的
#里面的权重并不是一开始全部为 0，而是会随机产生,可以让这些初始参数尽量保持一致
torch.manual_seed(seed)

# 判断当前电脑是否可以使用 CUDA 显卡
if torch.cuda.is_available():
    # 固定当前显卡产生的随机效果
    torch.cuda.manual_seed(seed)

    # 固定所有显卡产生的随机效果
    # 即使电脑只有一张显卡，这样写也没有问题
    torch.cuda.manual_seed_all(seed)
# 让cuDNN尽量选择确定性的计算方式
# 这样多次运行代码时，结果更容易保持一致
# cuDNN 是 NVIDIA 为深度学习计算提供的加速库，卷积操作经常会使用它
# 同一个卷积计算可能有多种实现方法,它们算出的结果理论上相同，但由于浮点数计算顺序不同，结果可能有极小误差
torch.backends.cudnn.deterministic = True
# 不让cuDNN每次自动寻找最快的卷积算法
# 因为自动选择算法有时会让多次训练结果产生细微变化
torch.backends.cudnn.benchmark = False

# 选择训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 输出当前使用的训练设备
print('当前训练设备',device)

# 如果可以使用显卡再输出显卡名称
if torch.cuda.is_available():
    print('显卡名称',torch.cuda.get_device_name(0))

# 定义 CIFAR-10 图片进入模型前的处理方式
# 下面先进行数据增强
# 训练集采用一套处理方式，验证集和测试集采用一套处理方式
# transform.Compose 图片会按从上到下的顺序进行处理
# 比如下面的 原始图片->RandomCrop->RandomHorizontalFlip->ToTensor->Normalize
# 训练集使用的图片的处理方式
train_transform = transforms.Compose([
    # 在图片四周先填充 4 个像素，由32 x 32 变成 40 x 40
    # 然后随机裁剪出一张 32 x 32的照片
    # 让图片中的物体位置发生轻微移动
    transforms.RandomCrop(32, padding=4),

    # 按一定概率把图片进行水平翻转
    # 例如：向右行驶的汽车可以被翻转成向左行驶
    # 这样不会改变图片类别，但可以让模型见到更多不同形式的图片
    transforms.RandomHorizontalFlip(),

    # 将图片转换成 pyTorch Tensor
    # 转换前通常是 PIL 图片
    # 普通图片顺序通常是【高度，宽度，通道】
    # 转换后形状变为：
    # 【通道数，高度，宽度】
    # CIFAR-10 是 RGB 彩色图片，所以形状是：
    # 【3，32，32】
    transforms.ToTensor(),

    # 对图片的三个通道分别进行标准化
    # 第一个元组是三个通道的均值：
    # （R通道均值，G通道均值，B通道均值）
    # 第二个元组是三个通道的标准差，同上
    # 使用 CIFAR-10 训练集计算得到的均值和标准差
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# 验证集和测试集使用的图片处理方式
valid_test_transform = transforms.Compose([
    transforms.ToTensor(),
    # 使用 CIFAR-10 训练集计算得到的均值和标准差
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    )
])

# 下载 CIFAR-10,并用Subset划分训练集和验证集
# 为什么要下载两份官方训练集
# 会把同一批训练下标放入第一份数据集，把验证下标放入第二份数据集
# 下载并读取 CIFAR-10 数据集

# 这份用于真正的模型训练，包含随机裁剪，随机水平翻转和标准化
full_train_dataset_aug = datasets.CIFAR10(
    # 数据集保存的位置
    # ‘./data’ 表示当前Notebook 所在目录下的data文件夹·
    root='./data',

    # train= true 表示读取 CIFAR-10 官方训练集
    train = True,
    download = True,
    transform  = train_transform
)

# 这个内容仍然是和上面一样，但是不会执行随机裁剪和随机水平翻转
full_train_dataset_plain = datasets.CIFAR10(
    # 和上面使用同一个数据保存位置
    root = './data',
    train = True,
    download = True,
    transform  =valid_test_transform
)

# 读取官方测试集
test_dataset = datasets.CIFAR10(
    root = './data',
    # train = false 表示官方测试集
    train = False,
    download  = True,
    transform = valid_test_transform
)

# 生成随机下标，方便后续subset发挥作用

# 获得官方训练集的总样本数量
total_size = len(full_train_dataset_aug)
valid_size = 5000
train_size = total_size - valid_size

# 创建一个pytorch 随机数生成器
generator = torch.Generator()

# 给这个随机数生成器设置固定随机种子
# 这样重新运行程序时，训练集和测试集的划分保持一致
generator.manual_seed(seed)

# 随机打乱全部下标
# torch.randperm(total_size) 会生成从0到 total_size-1的随机排列
indices = torch.randperm(
    total_size,
    generator = generator
)

# torch.randperm 返回的是Tensor
# 使用 .tolist()把他转成普通的Python列表
indices = indices.tolist()
# 把随机下标切成两部分，划分训练下标和验证下标

# 取随机下标列表中前45000个作为训练集下标
train_indices = indices[: train_size]
# 取剩下5000个作为验证集下标
valid_indices = indices[train_size:]

# 使用subset创建训练集和验证集
# Subset的基本格式就是 Subset(原始数据集，要保留的下标)

# 从带数据增强的完整训练集中，选取train_indices 对应的45000张照片‘
train_dataset  = Subset(
    full_train_dataset_aug,
    train_indices
)

# 从不带随机增强的完整训练集中选取对应的5000张照片
valid_dataset = Subset(
    full_train_dataset_plain,
    valid_indices
)

# 创建DataLoader
batch_size = 128

# 创建训练集DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    # shuffle = True 表示每个 epoch 开始前，
    # 都重新打乱训练样本的读取顺序
    shuffle=True,
    # num_workers 表示有多少个子进程负责读取数据
    num_workers = 0,
    # 如果使用CUDA pin_memory = True可以让CPU 数据传到 GPU 时更高效
    # 如果没有 CUDA,这里自动变成False
    pin_memory=torch.cuda.is_available()
)

# 创造验证集的DataLoader
valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# 创建测试集的DataLoader

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)
# 13. 从训练集 DataLoader 中取出一个 batch

# iter(train_loader)
# 把 train_loader 转换成一个可以逐批取数据的迭代器
# next(...)
# 从迭代器中取出第一个 batch
# images：这一批图片
# labels：这一批图片对应的标签
images, labels = next(iter(train_loader))


# 查看图片 batch 的形状
print("图片 batch 的形状：", images.shape)


# 查看标签 batch 的形状
print("标签 batch 的形状：", labels.shape)


# 查看图片的数据类型
print("图片的数据类型：", images.dtype)


# 查看标签的数据类型
print("标签的数据类型：", labels.dtype)

# 定义普通CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):

        # 调用父类nn.Module 的初始化方法
        super().__init__()

        # 第一组卷积
        # 输入【batch,3,32,32】
        # 输出【batch,32,16,16】

        self.conv_block1 = nn.Sequential(
            # 第一层卷积
            # in_channels = 3
            # 输入图片RGB有三个通道

            # out_channels =32
            # 使用 32 个 完整卷积核
            # 最终得到 32 张特征图

            # kernel_size = 3
            # 卷积核大小是 3x3

            #padding =1
            # 在图片四周填充一层像素
            # 这样卷积后的高度和宽度保持不变

            # 为什么 Covn2d 里面没有写batch_size
            # 因为 pyTorch 会自动识别输入的第0维是batch
            # 卷积层会使用同一组卷积核，分别处理这128张照片
            nn.Conv2d(3,32,3,1,1),

            # 对 32 个通道分别做归一化
            # 可以让中间特征的数值更加稳定
            nn.BatchNorm2d(32),
            # 引入非线性
            nn.ReLU(),

            # 再进行一次卷积
            # 输入通道已经变成了32
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 最大池化
            # kernel_size =2
            # 每次观察一个 2x2 的区域
            # stride 默认也是2
            # 所有宽度和高度也会缩小一半
            nn.MaxPool2d(2,2),
        )
        # 第二层卷积
        # 输入【batch,32,16,16】
        # 输出【batch,64,8,8】
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 再进行一次 64 通道卷积
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )
        # 第三层卷积
        # 输入【batch,64,8,8】
        # 输出【batch,128,4,4】

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2)
        )

        # 全局平均池化
        # 无论前面特征图的高度和宽度是多少
        # 都将每个通道压缩成 1x1

        # [batch,128,4,4]->[batch,128,1,1]
        self.avg_pool =nn.AdaptiveAvgPool2d(
            output_size=(1,1)
        )

        # 全局平均池化后，每张图片剩下128个特征
        # 将这128个特征映射成10个类别分数
        # 减少参数的同时，也降低一定的过拟合风险
        self.classifier = nn.Linear(128,10)

    # forward 定义数据进入模型后的传播顺序
    def forward(self,x):
        # 输入【batch,3,32,32】
        # 经过第一层卷积
        # 变成 【batch,32,16,16】
        x = self.conv_block1(x)
        # 经过第二组卷积
        # 【batch,32,16,16】-> [batch,64,8,8]
        x = self.conv_block2(x)

        # 经过第三组卷积
        # 【batch,64,8,8】 -> [batch,128,4,4]
        x= self.conv_block3(x)

        # 全局平均池化
        # 【batch,128,4,4】-> [batch,128,1,1]
        x = self.avg_pool(x)

        # 从一维开始展平
        # [batch,128,1,1] -> [batch,128]
        x = torch.flatten(x,start_dim=1)

        # 经过全连接分类层
        # 【batch,128】->[batch,10]
        x = self.classifier(x)

        return x

simple_cnn = SimpleCNN()
simple_cnn = simple_cnn.to(device)

# inception是属于并行结构，由于不清楚那种路线好，于是都走一遍，提取特征
#           |----->1x1 卷积-------             1x1：主要调整，组合通道
#           |                    |
#    输入 ----->1x1  -> 3x3 卷积 - |--->通道拼接  3x3 : 观察较小范围的局部特征
#           |                    |
#           |-->1x1  -> 5x5 卷积--|             5x5 ： 观察更大范围的局部特征
#           |                    |
#           |---池化 -> 1x1卷积----|             池化 ：保留比较明显的特征
# 定义 Inception基础块
class InceptionBlock(nn.Module):
    def __init__(
            self,
            # 输入特征图的通道数
            in_channels,

            # 第一条分支中 1x1 卷积输出的通道数
            out_1x1,

            # 第二条分支中：
            # 先用1x1卷积将通道数减少到 reduce_3x3
            reduce_3x3,

            # 再用3x3卷积输出out_3x3 个通道
            out_3x3,

            # 第三条分支中：
            # 先用1x1 卷积减少到 reduce_5x5个通道
            reduce_5x5,
            # 再用5x5卷积输出 out_5x5个通道
            out_5x5,

            # 第四条池化层最终输出通道数
            pool_proj
                 ):
        # 初始化父类 nn.Module
        super().__init__()

        # 分支1 ： 1x1卷积
        self.branch1 = nn.Sequential(
            # 1x1 卷积不会观察周围像素
            # 它主要是在每个位置上组合不同通道的信息
            # 【batch,inchannels,H,W】->[batch,out_1x1,H,W]
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_1x1,
                kernel_size=1
            ),
            # batchnorm 的参数必须等于卷积输出通道数
            nn.BatchNorm2d(out_1x1),
            nn.ReLU()
        )
        # 分支2 ： 1x1 卷积 + 3x3卷积
        self.branch2 = nn.Sequential(
            # 先用 1x1 卷积减少通道数
            # 如果直接使用 3x3 卷积
            # 输出通道很多时参数量会比较大
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = reduce_3x3,
                kernel_size=1
            ),
            nn.BatchNorm2d(reduce_3x3),
            nn.ReLU(),

            # 再用3x3 卷积提取局部特征
            # padding = 1可以让高度和宽度保持不变
            nn.Conv2d(
                in_channels = reduce_3x3,
                out_channels= out_3x3,
                kernel_size= 3,
                padding= 1
            ),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU()
        )

        # 分支3 : 1x1 卷积 + 5x5 卷积
        self.branch3 = nn.Sequential(
            # 先用1x1卷积降低通道数
            nn.Conv2d(
                in_channels =in_channels,
                out_channels=reduce_5x5,
                kernel_size= 1
            ),
            nn.BatchNorm2d(reduce_5x5),
            nn.ReLU(),

            # 5x5卷积可以观察到更大的局部区域
            # padding = 2 让卷积后的高度和宽度保持不变
            nn.Conv2d(
                in_channels = reduce_5x5,
                out_channels= out_5x5,
                kernel_size= 5,
                padding= 2
            ),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU()
        )

        # 分支4 ： 最大池化 + 1x1 卷积
        self.branch4 = nn.Sequential(
            # 使用 3x3 最大池化
            # stride =1
            # 池化窗口每次只移动一个像素
            # padding =1 保证池化前后的高度和宽度不变
            nn.MaxPool2d(
                kernel_size=3,
                stride = 1,
                padding =1
            ),
            # 池化后再用 1x1 卷积调整输出通道数
            nn.Conv2d(
              in_channels=in_channels,
              out_channels=pool_proj,
              kernel_size=1
            ),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU()
        )

    # 定义数据经过Inception 模块的传播方式
    def forward(self,x):
        # 同一份输入同时进入第一条分支
        branch1_output = self.branch1(x)

        # 同一份输入同时进入第二条分支
        branch2_output = self.branch2(x)

        # 同一份输入同时进入第三条分支
        branch3_output = self.branch3(x)

        # 同一份输入同时进入第四条分支
        branch4_output = self.branch4(x)

        # 将四个分支沿着通道维度拼接
        output = torch.cat([
            branch1_output,
            branch2_output,
            branch3_output,
            branch4_output
        ], dim=1)
        return output

# 定义适用于 CIFAR-10 的小型 InceptionNet
class InceptionNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 前置卷积层Stem
        # InceptionBlock 主要负责并行提取特征
        # 但输入原图只有3个通道
        # 所以先使用普通卷积提取一些基础特征
        # 比如输入【batch,3,32,32】 ->输出【batch,32,16,16】
        self.stem = nn.Sequential(
            # 第一次卷积：
            # 通道数从3 增加到 32
            nn.Conv2d(3,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 再进行一次 3x3 卷积
            # 输入通道输出通道都是32
            # 高宽不变
            nn.Conv2d(32,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 池化 【batch,32,32,32】->[batch,32,16,16]
            nn.MaxPool2d(kernel_size=2)
        )

        # 第一个InceptionBlock
        # 输入通道 32
        # 四个分支输出通道数：
        # 分支1：16
        # 分支2：24              这里是自定义的超参数
        # 分支3：8
        # 分支4：16
        self.inception1 = InceptionBlock(
            in_channels = 32,
            out_1x1 = 16,
            reduce_3x3 = 16,
            out_3x3 = 24,
            reduce_5x5 = 8,
            out_5x5 = 8,
            pool_proj = 16
        )
        # [batch,64,16,16]->[batch,128,16,16]
        # 第二个 InceptionBlock
        # inception1 输出了16+24+8+16 = 64个通道
        # 所以这里的in_channels =64
        # 让这里的四个分支输出 32+48+16+32 =128
        self.inception2 = InceptionBlock(
            in_channels = 64,
            out_1x1 = 32,
            reduce_3x3 = 24,
            out_3x3= 48,
            reduce_5x5 = 8,
            out_5x5 = 16,
            pool_proj = 32
        )
        # 第一次下采样
        #【batch,128,16,16】 ->[batch,128,8,8]
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # 第三个 inceptionblock
        # 输入通道128
        # 四个分支输出 48 + 72 + 24 + 48 =192
        self.inception3  = InceptionBlock(
            in_channels = 128,
            out_1x1 = 48,
            reduce_3x3 = 36,
            out_3x3= 72,
            reduce_5x5 = 12,
            out_5x5 = 24,
            pool_proj = 48
        )

        # 第四个InceptionBlock
        # 输入通道 192
        # 四个分支输出
        # 64 + 96 + 32 + 64 =256

        self.inception4 = InceptionBlock(
            in_channels = 192,
            out_1x1 = 64,
            reduce_3x3 = 48,
            out_3x3= 96,
            reduce_5x5 = 16,
            out_5x5 = 32,
            pool_proj = 64
        )

        # 第二次下采样
        # 【batch,256,8,8】 ->[batch,256,4,4]
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 全局平均池化
        # 将每个通道的 4x4 特征压缩成一个数字
        # [batch,256,4,4] -> [batch,256,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # 分类层
        # 全局平均池化后，每个照片剩256个特征
        # CIFAR-10有10个类别，所以输出10个logits

        self.classifier = nn.Linear(256,10)

    def forward(self, x):

        # 原始输入：[batch, 3, 32, 32]

        # 前置普通卷积
        # [batch, 3, 32, 32]->[batch, 32, 16, 16]
        x = self.stem(x)

        # 第一个 InceptionBlock
        # [batch, 32, 16, 16]-> [batch, 64, 16, 16]
        x = self.inception1(x)

        # 第二个 InceptionBlock
        # [batch, 64, 16, 16]->[batch, 128, 16, 16]
        x = self.inception2(x)

        # 第一次下采样
        # [batch, 128, 16, 16]->[batch, 128, 8, 8]
        x = self.pool1(x)


        # 第三个 InceptionBlock
        # [batch, 128, 8, 8]->[batch, 192, 8, 8]
        x = self.inception3(x)

        # 第四个 InceptionBlock
        # [batch, 192, 8, 8] ->[batch, 256, 8, 8]
        x = self.inception4(x)

        # 第二次下采样
        # [batch, 256, 8, 8]->[batch, 256, 4, 4]
        x = self.pool2(x)


        # 全局平均池化
        # [batch, 256, 4, 4]->[batch, 256, 1, 1]
        x = self.avg_pool(x)

        # 保留 batch 维度，
        # 将其余维度展平
        # [batch, 256, 1, 1]->[batch, 256]
        x = torch.flatten(
            x,
            start_dim=1
        )

        # 全连接分类
        # [batch, 256]->[batch, 10]
        x = self.classifier(x)

        return x
# 实例化 InceptionNet
inception_model = InceptionNet()
# 将模型移动到 GPU
inception_model = inception_model.to(device)

# 输入 x -----------------
#   |                    |
#  卷积                   |
#   |                    |
# BatchNorm              |
#   |                    |这
#  ReLU                  |个
#   |                    |叫
#  卷积                   |做
#   |                    |shortcut
# BatchNorm              |捷径连接
#   |                    |
# 得到F(x)                |
#   |                    |
# F(x)+x------------------
#   |
#  ReLU

# 定义Reset的基础残差块
# train_loader提供最初的照片，图片经过前面的层后，中间特征会作为x进入残差块
# 残差块让卷积分支学习变化量，再把原始输入加回来
# 残差块是在已有的基础上，还应该修改多少
class BasicBlock(nn.Module):
    def __init__(
            self,
            # 输入通道数
            in_channels,

            # 输出通道数
            out_channels,

            # 步长
            stride=1,
    ):
        super().__init__()
        #主分支第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride,
            # 保持卷积尺寸大小关系稳定
            padding =1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # 主分支第二层卷积
        self.conv2  = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding =1,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 这里先不马上写第二个RuLU,因为残差块通常先执行
        # out =F(x)_shortcut 再同一使用 ReLU(out)

        # 定义shortcut分支

        # 默认情况下，不需要对输入x进行处理
        # nn.Identity意思是输入什么就原样输出什么
        self.shortcut = nn.Identity()

        # 出现下面任意一种情况时，
        # 原始输入x和主分支输出的形状不同：
        # 1， stride 不等于1
        # 说明高度和宽度发生了变化
        # 2， in_channels 不等于 out_channels
        # 说明通道数发生了变化
        if(stride != 1 or in_channels != out_channels):
            # 使用 1x1 卷积调整通道数和空间尺寸
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    stride = stride,
                ),
                nn.BatchNorm2d(out_channels),
            )
    # 定义残差块的向前传播
    def forward(self, x):
        # 保存原始输入
        # shortcut 可能什么都不做
        # 也有可能用 1x1 卷积调整形状
        identity = self.shortcut(x)

        # 主分支
        # 第一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二层卷积
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加
        out += identity
        out = self.relu(out)
        return out

# 定义适用于 CIFAR-10 的小型ResNet
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 前置卷积层stem
        # [batch,3,32,32] -> [batch,32,32,32]
        self.stem = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # 每组放两个残差块的原因是：
        # 第一个块负责改变尺寸
        # 第二个块负责提取特征，但不在改变尺寸

        # 第一组残差块
        # [batch,32,32,32] ->[batch,32,32,32]
        # 通道数不变，高宽也不变

        self.layer1 = nn.Sequential(
            BasicBlock(32,32,1),
            BasicBlock(32,32,1)
        )
        # 第二组残差块
        # 第一个残差块 [batch,32,32,32] - > [batch,64,16,16]
        self.layer2 = nn.Sequential(
            # 通道数 32->64
            # 高宽 32x32 ->16x16
            BasicBlock(32,64,2),
            BasicBlock(64,64,1)
        )
        # 第三组残差块
        # [batch,64,16,16] -> [batch,128,8,8]
        self.layer3 = nn.Sequential(
            # 通道数 64 -> 128
            # 宽高 16x16 -> 8x8
            BasicBlock(64,128,2),

            # 通道高宽保持不变
            BasicBlock(128,128,1)
        )

        # 全局平均化
        # [batch,128,8,8] -> [batch,128,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        # 分类层
        self.classifier = nn.Linear(128,10)

    def forward(self,x):
        # 原始输入
        # [batch,3,32,32]

        # 前置卷积
        #[batch,3,32,32]->[batch,32,32,32]
        x = self.stem(x)

        # 第一组残差块
        # 【batch,32,32,32】-> [batch,32,32,32]
        x = self.layer1(x)

        # 第二组残差块
        # 【batch,32,32,32】 -> [batch,64,16,16]
        x = self.layer2(x)

        # 第三组残差块
        # [batch,64,16,16]->[batch,128,8,8]
        x = self.layer3(x)

        # 全局平均池化
        # [batch,128,8,8] ->[batch,128,1,1]
        x = self.avg_pool(x)

        # 展开乘二维
        # [batch,128,1,1] -> [batch,128]
        x = torch.flatten(x,start_dim=1)

        # 全连接分类
        # [batch,128] -> [batch,10]

        x= self.classifier(x)
        return x

# 实例化 ResNet
resnet_model = ResNet()

# 把模型移动到GPU
resnet_model = resnet_model.to(device)

# 把随机种子封装成函数
# 固定随机种子
def set_seed(seed = 42):
    # python随机数
    random.seed(seed)

    # Numpy随机数
    np.random.seed(seed)

    # pytorch CPU随机数
    torch.manual_seed(seed)

    # pytorch GPU随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 让卷积运算结果保持稳定
    #cuDNN 自动选择速度最快的算法，可能降低重复实验的一致性
    torch.backends.cudnn.benchmark = False
    #直接关闭了 cuDNN，卷积训练可能明显变慢。
    torch.backends.cudnn.enabled = True


# 通用训练函数
def train_model(
        model,
        model_name,
        train_loader,
        valid_loader,
        device,
        num_epochs = 20,
        learning_rate =0.001,
        weight_decay = 0.0005
):
    # 将当前模型移动到GPU
    model = model.to(device)

    # 当前模型使用的损失函数
    # CrossEntropyLoss 专门用于这种多分类任务
    criterion = nn.CrossEntropyLoss()

    # 当前模型使用自己独立的优化器
    # Adam 根据梯度更新模型中需要训练的参数
    # weight_decay = 0.0005 是权重衰减
    # 可在一定程度上抑制参数变得过大，降低过拟合风险
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

    # 保存每个epoch 的训练结果
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # 当前模型的最佳验证准确率
    best_valid_acc = 0.0

    # 当前模型的最佳参数
    best_model_state = None

    # 记录模型总训练时间
    total_start_time = time.time()

    print('='*70)
    print(f'开始训练:{model_name}')
    print('='*70)
    for epoch in range(num_epochs):
        # 记录当前 epoch 开始时间
        epoch_start_time = time.time()

        # 训练阶段
        model.train()

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        # 遍历训练集中所有batch
        for images,labels in train_loader :
            # 将当前batch移动到GPU
            images = images.to(device)
            labels = labels.to(device)

            # 清空上一个batch留下的梯度
#           # pytorch 会默认累加梯度
#           # 因此每个batch开始前都要把旧梯度清空
            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()
            # 当前batch的实际数量
            current_batch_size = images.size(0)

            #累计训练loss 总和
            train_loss_sum += (loss.item() * current_batch_size)
            # 得到预测类别
            pred = torch.argmax(outputs, dim=1)

            #累计预测正确数量
            train_correct += (pred == labels).sum().item()

            # 累计已处理的训练样本数量
            train_total += current_batch_size

        # 当前 epoch 的训练平均 loss
        train_epoch_loss = train_loss_sum / train_total
        # 当前epoch的准确率
        train_epoch_acc = train_correct / train_total

        # 验证阶段
        model.eval()

        valid_loss_sum = 0.0
        valid_correct = 0
        valid_total = 0


        # 验证阶段不需要：
        # loss.backward()
        # optimizer.step()
        # 因此关闭梯度记录，可以减少显存使用并提高速度
        with torch.no_grad():
         for images, labels in valid_loader:

            # 将当前验证 batch 移动到 GPU
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算当前验证 batch 的 Loss
            loss = criterion(outputs, labels)

            # 当前验证 batch 的图片数量
            current_batch_size = images.size(0)

            # 累计验证 Loss 总和
            valid_loss_sum += (
                loss.item() * current_batch_size
            )

            # 找到每张图片预测分数最大的类别
            predictions = outputs.argmax(dim=1)

            # 累计验证集中预测正确的数量
            valid_correct += (
                predictions == labels
            ).sum().item()

            # 累计已经处理的验证图片数量
            valid_total += current_batch_size

         valid_epoch_loss = valid_loss_sum / valid_total
         valid_epoch_acc = valid_correct/valid_total

         # 保存这一轮的训练记录
         # 把当前 epoch 的训练 Loss 加入列表
         train_losses.append(train_epoch_loss)

         # 把当前 epoch 的验证 Loss 加入列表
         valid_losses.append(valid_epoch_loss)

         # 把当前 epoch 的训练准确率加入列表
         train_accuracies.append(train_epoch_acc)

         # 把当前 epoch 的验证准确率加入列表
         valid_accuracies.append(valid_epoch_acc)

         # 保存最佳模型
         # 如果当前验证准确率超过以前的最佳结果
         if valid_epoch_acc > best_valid_acc:

            # 更新最佳验证准确率
            best_valid_acc = valid_epoch_acc

            # 复制当前模型参数
            # deepcopy 会保存一份独立的参数快照
            # 后续模型继续训练时，不会修改这份最佳参数
            best_model_state = copy.deepcopy(
            model.state_dict()
            )
         # 输出当前epoch 结束时间
         epoch_end_time = time.time()
         epoch_time = epoch_end_time - epoch_start_time
         print(f'Epoch [{epoch+1}/{num_epochs}] ')
         print(f'Train Loss : {train_epoch_loss:.4f} |'
              f'Train Accuracy : {train_epoch_acc:.4f} ')

         print(f'Valid Loss : {valid_epoch_loss:.4f} |'
              f'Valid Accuracy : {valid_epoch_acc:.4f} ')
         print(f'Epoch Time: {epoch_time:.2f} 秒')

         # 所有 epoch 训练结束

    total_training_time = time.time() - total_start_time

    # 将最佳模型加载回当前模型
    model.load_state_dict(best_model_state)

    # 根据模型名称自动生成文件名
    file_name = (model_name.lower().replace(' ','_')+'_best.pth')

    # 保存最佳模型参数
    torch.save(best_model_state, file_name)
    print(f'{model_name}训练完成！')

    # 当前模型的训练历史
    # 用字典将当前的数据保存下来便于后续画图比较
    history = {

            "train_loss": train_losses,

            "valid_loss": valid_losses,

            "train_acc": train_accuracies,

            "valid_acc": valid_accuracies,

            "best_valid_acc": best_valid_acc,

            "training_time": total_training_time
        }

    return model,history
# 删除前面用于测试模型结构的对象
del simple_cnn
del inception_model
del resnet_model

# 清理没有继续使用的显存缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print('测试模型已经清理')
# 三个模型的创建方法
model_builders = {
    'Simple CNN' : SimpleCNN,
    'Inception' : InceptionNet,
    'ResNet': ResNet,
}

# 保存训练完成模型
trained_models = {}

# 保存三个模型的训练记录
all_histories ={}

# 自动依次训练三个模型

for model_name, model_builder in model_builders.items():

    # 每个模型训练前重新固定随机种子
    set_seed(seed)


    # 根据模型类创建一个全新的模型
    model = model_builder()


    # 调用同一个训练函数
    trained_model, history = train_model(

        model=model,

        model_name=model_name,

        train_loader=train_loader,

        valid_loader=valid_loader,

        device=device,

        num_epochs=20,

        learning_rate=0.001,

        weight_decay=0.0005
    )


    # 将训练完成的模型移动回 CPU
    #
    # 避免它继续占用 GPU 显存
    trained_model = trained_model.to("cpu")


    # 保存训练完成的模型
    trained_models[model_name] = trained_model


    # 保存当前模型的训练历史
    all_histories[model_name] = history


    # 删除临时模型变量
    del model
    del trained_model


    # 清理未使用的 GPU 显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 绘制三个模型的对比曲线

# 从其中一个模型的 history 中获取 epoch 数
# 因为三个模型都训练了相同的轮数，所以取谁都可以
num_epochs = len(all_histories["Simple CNN"]["train_loss"])

# 生成横坐标：1, 2, 3, ..., num_epochs
epochs = range(1, num_epochs + 1)


# 1. 训练集 Loss 对比
plt.figure(figsize=(8, 5))

for model_name, history in all_histories.items():
    plt.plot(
        epochs,
        history["train_loss"],
        label=model_name
    )

plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss Comparison")
plt.legend()
plt.grid(True)
plt.show()


# 2. 验证集 Loss 对比

plt.figure(figsize=(8, 5))

for model_name, history in all_histories.items():
    plt.plot(
        epochs,
        history["valid_loss"],
        label=model_name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Comparison")
plt.legend()
plt.grid(True)
plt.show()


# 3. 训练集 Accuracy 对比

plt.figure(figsize=(8, 5))

for model_name, history in all_histories.items():
    plt.plot(
        epochs,
        history["train_acc"],
        label=model_name
    )

plt.xlabel("Epoch")
plt.ylabel("Train Accuracy")
plt.title("Train Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.show()


# 4. 验证集 Accuracy 对比
plt.figure(figsize=(8, 5))

for model_name, history in all_histories.items():
    plt.plot(
        epochs,
        history["valid_acc"],
        label=model_name
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy Comparison")
plt.legend()
plt.grid(True)
plt.show()

# 测试集评估函数
def evaluate_model(
        model,
        test_loader,
        device
):
    # 将模型移动到GPU
    model = model.to(device)

    # 切换测试模式
    # BatchNorm会使用训练阶段保存的均和方差
    model.eval()

    # 测试集也使用交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 累计测试集loss
    test_loss_sum = 0.0

    # 累计预测正确的图片数量
    test_correct = 0

    # 累计测试图片数量
    test_total = 0

    # 测试阶段不计算梯度
    with torch.no_grad():
        # 遍历官方测试集
        for images,labels in test_loader:

            # 将图片和标签移动到GPU
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs,labels)

            # 当前 batch 的实际图片数量
            current_batch_size = images.size(0)

            # 累计测试loss总和
            test_loss_sum += loss.item()*current_batch_size

            # 找出每张图片预测分数最大的类别
            predictions = outputs.argmax(dim=1)

            # 累计预测正确数量
            test_correct += (predictions == labels).sum().item()

            # 累计测试图片总数
            test_total += current_batch_size
    # 测试集平均loss
    test_loss = test_loss_sum / test_total
    # 测试集准确率
    test_accuracy = test_correct / test_total

    #将模型放回CPU,释放GPU显存
    model = model.to('cpu')

    return test_loss, test_accuracy

# 使用官方测试集评估三个模型
test_results = {}

for model_name,model in trained_models.items():
    # 测试当前模型
    test_loss, test_accuracy = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )
    # 保存测试结果
    test_results[model_name] ={
        'test_loss' : test_loss,
        'test_accuracy' : test_accuracy
    }

    print('='*50)
    print(f"模型：{model_name}")
    print(f"Test Loss：{test_loss:.4f}")
    print(f"Test Accuracy：{test_accuracy:.4f}")

    # 清理未使用的 GPU 显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("=" * 50)

# 测试集准确率折线图


model_names = list(
    test_results.keys()
)

test_accuracies = [
    test_results[model_name]["test_accuracy"]
    for model_name in model_names
]


plt.figure(figsize=(8, 5))

plt.plot(
    model_names,
    test_accuracies,
    marker="o",
    linewidth=2
)


# 在每个点上方显示具体准确率
for model_name, accuracy in zip(
    model_names,
    test_accuracies
):
    plt.text(
        model_name,
        accuracy + 0.002,
        f"{accuracy:.4f}",
        ha="center"
    )


plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Comparison")

plt.grid(True)
plt.show()

# 测试集 Loss 折线图


model_names = list(
    test_results.keys()
)

test_losses = [
    test_results[model_name]["test_loss"]
    for model_name in model_names
]


plt.figure(figsize=(8, 5))

plt.plot(
    model_names,
    test_losses,
    marker="o",
    linewidth=2
)


# 显示每个模型的具体测试 Loss
for model_name, loss in zip(
    model_names,
    test_losses
):
    plt.text(
        model_name,
        loss + 0.005,
        f"{loss:.4f}",
        ha="center"
    )


plt.xlabel("Model")
plt.ylabel("Test Loss")
plt.title("Test Loss Comparison")

plt.grid(True)
plt.show()

# =========================================================
# 最佳验证准确率和测试准确率对比
# =========================================================

model_names = list(
    all_histories.keys()
)


best_valid_accuracies = [
    all_histories[model_name]["best_valid_acc"]
    for model_name in model_names
]


test_accuracies = [
    test_results[model_name]["test_accuracy"]
    for model_name in model_names
]


plt.figure(figsize=(9, 5))


# 最佳验证准确率
plt.plot(
    model_names,
    best_valid_accuracies,
    marker="o",
    linewidth=2,
    label="Best Validation Accuracy"
)


# 测试集准确率
plt.plot(
    model_names,
    test_accuracies,
    marker="o",
    linewidth=2,
    label="Test Accuracy"
)


plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title(
    "Validation and Test Accuracy Comparison"
)

plt.legend()
plt.grid(True)
plt.show()