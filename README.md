# TrackS - 基于GNN的焊缝跟踪系统

## 简介
论文

## 对比
### 对比方法1--方向模板法
[DOI: 10.1109/JSEN.2018.2824660](https://ieeexplore.ieee.org/document/8333759)

[对比方法1代码](https://github.com/zhhusst/trackS/blob/master/GNNTransformer/%E5%AF%B9%E6%AF%94%E6%96%B9%E6%B3%951/%E5%AF%B9%E6%AF%94%E6%96%B9%E6%B3%951.py)

### 对比方法2--基于规则的方法
[对比方法2代码](https://github.com/zhhusst/trackS/blob/master/GNNTransformer/%E5%AF%B9%E6%AF%94%E6%96%B9%E6%B3%951/%E5%AF%B9%E6%AF%94%E6%96%B9%E6%B3%951.py)

### 对比方法3--自定义卷积核

#### 论文原文
    [DOI: 10.1016/j.rcim.2021.102279](https://doi.org/10.1016/j.rcim.2021.102279)

#### 论文复现代码
   [对比方法3代码](https://github.com/zhhusst/trackS/blob/master/GNNTransformer/%E5%AF%B9%E6%AF%94%E6%96%B9%E6%B3%953/%E5%AF%B9%E6%AF%94%E6%96%B9%E6%B3%953.py)


#### 测试结果

**回归率** 
    
    测试图像总数: 39
    成功检测的图像数: 31
    未检测到的图像数: 8

**欧式距离（像素误差）统计:**
    
    平均值: 76.33 像素
    中位数: 3.00 像素
    最小值: 1.00 像素
    最大值: 492.71 像素

**X坐标误差统计:**
   
    MAE: 75.13 像素
    RMSE: 169.01 像素

**Y坐标误差统计:**
   
    MAE: 9.06 像素
    RMSE: 18.84 像素

**总结:**

    该方法虽然是自定义卷积核，但是其卷积核设计过分依赖于先验知识，比如说对与焊缝形状的要求过于严格，每次遇到不同的焊缝形状都需要重新设计卷积核。对于相同形状的焊缝其检测结果不错。

### 对比方法4--基于语义的方法

#### 论文原文
    [DOI: 10.1109/TIE.2017.2694399](https://ieeexplore.ieee.org/document/7903643)

#### 论文复现代码
    [对比方法4代码](https://github.com/zhhusst/trackS/blob/82ea9db838e656c024ff9555932c2ba9e143c644/GNNTransformer/%E5%AF%B9%E6%AF%94%E6%96%B9%E6%B3%954/Automatic_Welding_Seam_Tracking_and_Identificatio.py)

## 本文方法
每一份代码中均涵盖同样的测试内容。

1. 下载数据集，并解压到代码同级目录下，确保路径为`./dataset`。
   数据集下载链接: [百度网盘链接](https://pan.baidu.com/s/1pLhGZ5jv4nX4F3Ykz7Jf1A?pwd=abcd) 提取码: abcd
2. 安装所需的Python库，建议使用Python 3.7及以上

### 实验结果

**超参数设置**
    
    [超参数设置](GNNTransformer/log/training_logs_20250916-222100/hyperparameters.json)

**回归率** 

    测试图像总数: 39
    成功检测的图像数: 39
    未检测到的图像数: 0

**欧式距离（像素误差）统计**
    
    平均值: 3.77 像素
    中位数: 1.88 像素
    最小值: 0.21 像素
    最大值: 20.34 像素

**X坐标误差统计**
    
    MAE: 3.48 像素
    RMSE: 5.36 像素

**Y坐标误差统计**
  
    MAE: 0.90 像素
    RMSE: 1.52 像素
