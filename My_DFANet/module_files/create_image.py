import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
# from demo import interaction_net
import os
import matplotlib.pyplot as plt


# dataset index :trento, muufl, houston2013, houston2018
dataset_index = 1


hsi_data=[]
lidar_data=[]
gt_hsi=[]
gt_=[]
if dataset_index == 1:
    print("The dataset is trento")
    trento_hsi_data = np.load('../../data_Trento_Muufl/HSI_Trento_600_166_63.npy')
    trento_gt = np.load('../../data_Trento_Muufl/GT_Trento_600_166_63.npy')
    trento_lidar_data = np.load('../../data_Trento_Muufl/Lidar1_Trento_600_166_63.npy')#1张
    hsi_data = trento_hsi_data
    lidar_data = trento_lidar_data
    gt_hsi = trento_gt

    gt_ = gt_hsi


elif dataset_index == 2:
    print("The dataset is muufl")
    muufl_hsi_data = np.load('../../data_Trento_Muufl/muufl_hsi_325_220_64.npy')
    muufl_gt = np.load('../../data_Trento_Muufl/muufl_new_gt.npy')
    muufl_lidar_data = np.load('../../data_Trento_Muufl/muufl_lidar_firstandlastreturn_325_220_2.npy')#2张
    # muufl_lidar_data = np.load('TrentoandMUUFLnpy/muufl_lidarw2_325_220_64.npy')
    hsi_data = muufl_hsi_data
    lidar_data = muufl_lidar_data
    gt_hsi = muufl_gt

    gt_ = gt_hsi


elif dataset_index == 3:
    print("The dataset is houston2013")
    houston13_hsi_data = np.load('../../data_houston2013/Houston2013_hsi.npy')
    houston13_gt = np.load('../../data_houston2013/Houston2013_gt.npy')
    houston13_lidar_data = np.load('../../data_houston2013/Houston2013_lidar.npy')#1张
    hsi_data = houston13_hsi_data
    lidar_data = houston13_lidar_data
    gt_hsi = houston13_gt

    gt_ = gt_hsi


elif dataset_index == 4:
    print("The dataset is houston2018")
    houston18_hsi_data = np.load('../../data_houston2018/downsampled_Houstonhsi.npy')
    houston18_gt = np.load('../../data_houston2018/downsampled_gt.npy')
    houston18_lidar_data = np.load('../../data_houston2018/downsampled_lidardata.npy')#3张
    hsi_data = houston18_hsi_data
    lidar_data = houston18_lidar_data
    gt_hsi = houston18_gt

    gt_ = gt_hsi


# parameters
data_x, data_y, band = hsi_data.shape[0], hsi_data.shape[1], hsi_data.shape[2]
lidar_band = lidar_data.shape[2]

# 示例分类数据，假设3x3的分类结果
classification_map = gt_hsi
# 定义类别和颜色字典
# trento
color_dict = {
    0: '#000000',
    1: '#00007E',
    2: '#0000FE',
    3: '#80FF00',
    4: '#FFFF00',
    5: '#FE0000',
    6: '#7E0001',
}
# muufl
# color_dict = {
#     0: '#000000',
#     1: '#077512',
#     2: '#15ff27',
#     3: '#1fffff',
#     4: '#fdc61d',
#     5: '#fd092b',
#     6: '#1200c3',
#     7: '#5c00c3',
#     8: '#fd758e',
#     9: '#c45c0b',
#     10: '#feff27',
#     11: '#c41a5a',
# }
# Houston2013
# color_dict = {
#     0: '#000000',  # 黑色
#     1: '#0fb11c',  # 亮绿色
#     2: '#067112',  # 深绿色
#     3: '#0b7f3f',  # 草绿色
#     4: '#023d06',  # 极深绿色
#     5: '#a37416',  # 黄棕色
#     6: '#16b6ba',  # 亮青色
#     7: '#6c0200',  # 深红色
#     8: '#d6d6f5',  # 淡紫色
#     9: '#6e6e6e',  # 灰色
#     10: '#c8a57a',  # 浅棕色
#     11: '#d5a973',  # 杏色
#     12: '#585858',  # 深灰色
#     13: '#b0a95a',  # 橄榄绿色
#     14: '#14eb24',  # 明亮绿色
#     15: '#c71631',  # 鲜红色
# }

# 创建一个自定义的颜色列表
cmap_list = [color_dict[i] for i in sorted(color_dict.keys())]
# 创建自定义的colormap
cmap = mcolors.ListedColormap(cmap_list)
# 定义颜色边界
bounds = np.arange(len(color_dict) + 1) - 0.5
norm = mcolors.BoundaryNorm(bounds, cmap.N)
# 显示分类地图
plt.imshow(classification_map, cmap=cmap, interpolation='none')
# 创建图例
legend_patches = [mpatches.Patch(color=color_dict[i], label=f'Class {i}') for i in sorted(color_dict.keys())]
plt.legend(handles=legend_patches, loc='lower center', ncol=(len(color_dict)//2)+1, title='Classes', bbox_to_anchor=(0.5, -0.05))
plt.title('Hyperspectral Image Classification Map')
plt.axis('off')

output_path = './' + str(dataset_index) + '925tuli.png'  # 指定保存的路径和文件名
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()

print('over')