# GeoC_lenet
基于lenet网络实现岩屑薄片分类，网络结构搭建好了，两层卷积，两层全连接，体会到了电脑性能的重要性了，到现在一遍还没跑完。就2000张图片就这样了，还咋训练啊。。。

# 数据处理
## get_Data_from_doc.py 提取doc文件中的图片和岩性信息
## build_Geo_data.py 处理图片文件为tfrecord格式


# 构建网络
## GeoC_lenet_forward.py 向前传播
## GeoC_lenet_backward.py 向后传播

# 工具函数
## utils.py 读入数据等工具函数
