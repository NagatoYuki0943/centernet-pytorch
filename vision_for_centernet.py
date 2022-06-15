"""
特征点预测
"""

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # 模型的输出是 128,128,64
    height, width, feat_stride = 128,128,1

    fig = plt.figure(figsize=(16, 7))

    """原图"""
    ax  = fig.add_subplot(121)  # 1行2列第1个
    plt.ylim(-10,17)    # y轴范围
    plt.xlim(-10,17)    # x轴范围

    # 生成 128*128网格
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 堆叠2次x,y坐标                                            -1 指 2
    boxes   = np.stack([shift_x,shift_y,shift_x,shift_y],axis=-1).reshape([-1,4]).astype(np.float32)
    # print(boxes.shape)  # (16384, 4)

    # 画除了前3个点的其他点
    plt.scatter(boxes[3:,0],boxes[3:,1])
    # 画3个点
    plt.scatter(boxes[0:3,0],boxes[0:3,1],c="r")
    ax.invert_yaxis()

    """调整后的图片"""
    ax = fig.add_subplot(122)   # 1行2列第2个
    plt.ylim(-10,17)
    plt.xlim(-10,17)

    # 生成 128*128网格
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # 堆叠2次x,y坐标
    boxes   = np.stack([shift_x,shift_y,shift_x,shift_y],axis=-1).reshape([-1,4]).astype(np.float32)
    # 画全部点
    plt.scatter(shift_x,shift_y)

    # 随机生成热力图(不绘制,没法绘制) 0~1之间               80分类
    heatmap = np.random.uniform(0,1,[128,128,80]).reshape([-1,80])
    # 随机生成位置调整                                     2个参数
    reg     = np.random.uniform(0,1,[128,128,2]).reshape([-1,2])
    # 随机生成宽高调整                                     2个参数
    wh      = np.random.uniform(5,20,[128,128,2]).reshape([-1,2])

    # 调整中心位置
    boxes[:,:2] = boxes[:,:2] + reg
    boxes[:,2:] = boxes[:,2:] + reg
    plt.scatter(boxes[0:3,0],boxes[0:3,1])

    # 调整宽高
    boxes[:,:2] = boxes[:,:2] - wh/2
    boxes[:,2:] = boxes[:,2:] + wh/2

    # 绘制3次
    for i in [0,1,2]:
        rect = plt.Rectangle([boxes[i, 0],boxes[i, 1]], wh[i,0], wh[i,1], color="r",fill=False)
        ax.add_patch(rect)

    ax.invert_yaxis()

    plt.show()
