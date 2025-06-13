import os
import time
import random
import numpy as np
import torch

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from Utils.generate_pic_idx import sampling_num
from sklearn.preprocessing import StandardScaler
from torch import optim
import torch.nn.functional as F
from sklearn import metrics
from Utils.InfoNCE import InfoNCE
from datetime import datetime

from My_DFANet.module_files.function_file import patch_handle, aa_and_each_accuracy
from My_DFANet.module_files.configs_file import config_trento, config_muufl, config_houston2018, config_houston2013

from net_6_19 import DFANet
from skimage.filters import sobel

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
InfoNCE = InfoNCE(negative_mode='paired')

def loss(x, y, weight):
    criterion = F.cross_entropy
    x = x.to(device)
    y = y.to(device)
    weight = weight.to(device)
    losses = criterion(x, y.long() - 1, weight=None, ignore_index=-1, reduction='none').to(device)
    # losses = focal_loss(x, y.long() - 1).to(device)
    # print(losses)

    v = losses.mul_(weight).sum() / weight.sum()
    return v

def cosine_similarity_loss(output, target):
    # 确保输出和目标张量形状一致
    if output.shape != target.shape:
        raise ValueError(
            f"Output and target shapes must match. Output shape: {output.shape}, Target shape: {target.shape}")
    return 1 - F.cosine_similarity(output, target).mean()

# 定义一个可以设置随机种子的函数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main_functions(num, ep):

    # dataset index :trento, muufl, houston2013, houston2018
    dataset_index = 1

    for i in range(num):

        # seed_list = [814 641] #种子 muufl 867
        # seed = random.randint(0, 1000)
        seed = 656 # 571 641(3dataset0.9580) 987    muufl：656 0.9285
        # 设置随机数种子
        setup_seed(seed)
        print('seed: ', seed)

        hsi_data=[]
        lidar_data=[]
        gt_hsi=[]
        gt_=[]
        if dataset_index == 1:
            print("The dataset is trento")
            trento_hsi_data = np.load('../data_Trento_Muufl/HSI_Trento_600_166_63.npy')
            trento_gt = np.load('../data_Trento_Muufl/GT_Trento_600_166_63.npy')
            trento_lidar_data = np.load('../data_Trento_Muufl/Lidar1_Trento_600_166_63.npy')#1张
            hsi_data = trento_hsi_data
            lidar_data = trento_lidar_data
            gt_hsi = trento_gt

            gt_ = gt_hsi


        elif dataset_index == 2:
            print("The dataset is muufl")
            muufl_hsi_data = np.load('../data_Trento_Muufl/muufl_hsi_325_220_64.npy')
            muufl_gt = np.load('../data_Trento_Muufl/muufl_new_gt.npy')
            muufl_lidar_data = np.load('../data_Trento_Muufl/muufl_lidar_firstandlastreturn_325_220_2.npy')#2张
            # muufl_lidar_data = np.load('TrentoandMUUFLnpy/muufl_lidarw2_325_220_64.npy')
            hsi_data = muufl_hsi_data
            lidar_data = muufl_lidar_data
            gt_hsi = muufl_gt

            gt_ = gt_hsi


        elif dataset_index == 3:
            print("The dataset is houston2013")
            houston13_hsi_data = np.load('../data_houston2013/Houston2013_hsi.npy')
            houston13_gt = np.load('../data_houston2013/Houston2013_gt.npy')
            houston13_lidar_data = np.load('../data_houston2013/Houston2013_lidar.npy')#1张
            hsi_data = houston13_hsi_data
            lidar_data = houston13_lidar_data
            gt_hsi = houston13_gt

            gt_ = gt_hsi


        elif dataset_index == 4:
            print("The dataset is houston2018")
            houston18_hsi_data = np.load('../data_houston2018/downsampled_Houstonhsi.npy')
            houston18_gt = np.load('../data_houston2018/downsampled_gt.npy')
            houston18_lidar_data = np.load('../data_houston2018/downsampled_lidardata.npy')#3张
            hsi_data = houston18_hsi_data
            lidar_data = houston18_lidar_data
            gt_hsi = houston18_gt

            gt_ = gt_hsi


        # test edge patch
        single_band = hsi_data[:,:,0]
        sobel_edge = sobel(single_band)
        sobel_edge_normalized = (sobel_edge - sobel_edge.min()) / (sobel_edge.max() - sobel_edge.min())
        # 设定阈值，将其转换为 0-1 矩阵
        threshold = 0.1
        binary_edge = (sobel_edge_normalized > threshold).astype(np.uint8)
        # 显示第一个波段的原始图像和 Sobel 滤波结果
        plt.imshow(binary_edge, cmap='gray')
        plt.title('Original Band 0')
        plt.axis('off')

        output_path = './' + str(dataset_index) + '133325.png'  # 指定保存的路径和文件名
        # plt.savefig(output_path, dpi=800, bbox_inches='tight')
        # plt.show()

        # parameters
        data_x, data_y, band = hsi_data.shape[0], hsi_data.shape[1], hsi_data.shape[2]
        lidar_band = lidar_data.shape[2]
        # lidar_band = 2

        gt_x, gt_y = gt_hsi.shape[0], hsi_data.shape[1]

        # select samples
        gt_hsi_ = gt_.reshape(-1)
        train_id, test_id, class_num, select_num, test_num = sampling_num(gt_hsi_, dataset_index)#注意不要过拟合
        # train_id, test_id, class_num, select_num, test_num = sampling_pro(0.95, gt_hsi_)#注意不要过拟合
        all_labels_num = sum(class_num)
        # 保存样本采样掩码
        # mask_zero = np.zeros(gt_hsi_.size)
        # train_mask = mask_zero.copy()
        # train_mask[train_id] = 1
        # train_mask = train_mask.reshape(gt_x, gt_y).astype(int)
        # train_mask = train_mask * gt_hsi
        #
        # test_mask = mask_zero.copy()
        # test_mask[test_id] = 1
        # test_mask = test_mask.reshape(gt_x, gt_y).astype(int)
        # test_mask = test_mask * gt_hsi
        # np.save('../houston2013/train_mask.npy', train_mask)
        # np.save('../houston2013/test_mask.npy', test_mask)

        print('The number of selected training samples:', select_num)
        print('The number of selected testing samples:', test_num)

        # mask_train and mask_test
        # train_id = np.load('C:\\Users\Administrator\Desktop\code\\newFPGA\index.npy')

        # 测试集标签
        test_gt = gt_hsi_[test_id]

        # data = preprocessing.scale(data)
        data = hsi_data.reshape((data_x*data_y, band))
        data_lidar = lidar_data.reshape((data_x*data_y, lidar_band))
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        data_lidar = scaler.fit_transform(data_lidar)

        data_ = data.reshape((data_x, data_y, band))
        data_lidar_ = data_lidar.reshape((data_x, data_y, lidar_band))
        gt_ = gt_hsi.reshape((gt_x, gt_y))

        print('making dataset')
        # 切片处理
        patch_length = 4
        batch_size = 512
        train_iter, test_iter, all_iter = patch_handle(data_, data_lidar_, train_id, test_id, gt_, patch_length, batch_size, dataset_index)

        # net
        net = []
        if dataset_index == 1:
            net = DFANet(config_trento['config'])
        elif dataset_index == 2:
            net = DFANet(config_muufl['config'])
        elif dataset_index == 3:
            net = DFANet(config_houston2013['config'])
        elif dataset_index == 4:
            net = DFANet(config_houston2018['config'])

        # 读取网络参数
        current_date = datetime.now().strftime("%Y%m%d")
        dataset_index = dataset_index  # 假设数据集索引为 1
        # checkpoint_path = f"net_pth/model_{dataset_index}_{current_date}.pth"
        # try:
        #     checkpoint = torch.load(checkpoint_path)
        #     net.load_state_dict(checkpoint, strict=True)
        #     print("模型参数已成功加载。")
        # except FileNotFoundError:
        #     print(f"找不到文件：{checkpoint_path}")
        # except RuntimeError as e:
        #     print(f"加载模型时出错：{e}")

        net = net.to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4, nesterov=True)
        # 定义学习率调度器，每200个epoch将学习率衰减到原来的0.1倍
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

        criterion = torch.nn.CrossEntropyLoss().to(device)

        distill_optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4, nesterov=True)

        max_point_acc = 0
        output_class_total = []
        print('start training')
        Epoch = ep
        for epoch in range(Epoch):
            train_loss_sum, train_acc_sum, n = 0.0, 0.0, 0
            con_loss_sum = 0.0
            batch_count = 0
            time_o = time.time()

            net.train()
            for x, y, z in train_iter:
                x = x.to(device)
                y = y.to(device)
                z = z.to(device)

                logits, labels, loss_d = net(x, y) # 原来的
                # 自蒸馏损失
                # scaler = GradScaler(enabled=True)
                # dis_loss = loss_d
                # distill_optimizer.zero_grad()
                #
                # scaler.scale(dis_loss).backward(retain_graph=True)
                # scaler.step(distill_optimizer)
                # scaler.update()
                # print('dis_loss:', dis_loss.item())

                # 分类结果
                output_class = labels.argmax(dim=1) + 1 # output属于哪一类
                loss_train = criterion(labels, z.long()-1) # 5*loss_d
                # loss_train = loss_train + 50 * loss_d
                loss_train = loss_train

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

                # 更新学习率
                # scheduler.step()

                train_loss_sum += loss_train.cpu().item()
                train_acc_sum += (output_class == z).sum().cpu().item()
                # con_loss_sum += con_loss.cpu().item()
                n += y.shape[0]
                batch_count += 1

            oa = (train_acc_sum / n)
            total_loss = (train_loss_sum / batch_count)
            conloss_t = (con_loss_sum / batch_count)
            time_e = time.time()
            train_time = time_e - time_o
            print('Epoch:', epoch, 'oa:', "%.4f" % oa, 'Loss:', "%.4f" % total_loss, 'train_time(s):', "%.2f" % train_time)

            if epoch%20 == 0 or epoch == (Epoch-1):
                print('-----------------------------------------------------')
                print('start testing')

                test_loss, test_acc, n = 0.0, 0.0, 0
                batch_count = 0
                test_time_o = time.time()
                output_class_total = []
                current_point_acc = 0

                net.eval()
                with torch.no_grad():
                    for x, y, z in test_iter:
                        x = x.to(device)
                        y = y.to(device)
                        z = z.to(device)

                        # pse_out, feat_out = pse_net(x, y)
                        # # # 对比损失
                        # pse_loss = cosine_similarity_loss(pse_out, feat_out)
                        # pse_loss_show = pse_loss.cpu().item()
                        # # print('pseudo_loss:', "%.4f" % pse_loss_show)
                        #
                        # with torch.no_grad():
                        #     pse_out = pse_out.clone()
                        _, output, _ = net(x, y)
                        # _, output, _ = simclr_net(x, y, augmented_x, augmented_y, 1)# is_test=1
                        output_class_t = output.argmax(dim=1) + 1  # output属于第几类

                        loss_test = criterion(output, z.long() - 1)

                        test_loss += loss_test.cpu().item()
                        test_acc += (output_class_t == z).sum().cpu().item()
                        n += y.shape[0]
                        batch_count += 1

                        output_class_total.extend(np.array(output_class_t.cpu()))

                test_oa = (test_acc / n)
                test_avg_l = (test_loss / batch_count)
                test_time_e = time.time()
                test_time = test_time_e - test_time_o
                print('trained epoch:', epoch, 'oa:', "%.4f" % test_oa, 'Loss:', "%.4f" % test_avg_l, 'test_time(s):', "%.2f" % test_time)

                # 保存网络参数
                current_point_acc = test_oa
                if current_point_acc > max_point_acc:
                    max_point_acc = current_point_acc

                    # 获取当前日期并格式化为字符串
                    current_date = datetime.now().strftime("%Y%m%d")
                    file_name = f"model_{dataset_index}"+f"_{current_date}"+".pth"
                    # 假设 net_path 是文件夹路径
                    net_path = "net_pth"
                    # 拼接文件夹路径和文件名
                    full_path = os.path.join(net_path, file_name)
                    # 创建文件夹（如果不存在）
                    os.makedirs(net_path, exist_ok=True)

                    torch.save(net.state_dict(), full_path)
                    print(f"Model saved to {full_path}")


        print('last test')
        # 读取模型
        checkpoint_path = f"net_pth/model_{dataset_index}_{current_date}.pth"
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            net.load_state_dict(checkpoint, strict=True)
            print("模型参数已成功加载。")
        except FileNotFoundError:
            print(f"找不到文件：{checkpoint_path}")
        except RuntimeError as e:
            print(f"加载模型时出错：{e}")


        net.eval()
        output_class_total = []
        with torch.no_grad():
            for x, y, z in test_iter:
                x = x.to(device)
                y = y.to(device)
                z = z.to(device)

                # pse_out, feat_out = pse_net(x, y)
                # # # 对比损失
                # pse_loss = cosine_similarity_loss(pse_out, feat_out)
                # pse_loss_show = pse_loss.cpu().item()
                # # print('pseudo_loss:', "%.4f" % pse_loss_show)
                #
                # with torch.no_grad():
                #     pse_out = pse_out.clone()
                _, output, _ = net(x, y)
                # _, output, _ = simclr_net(x, y, augmented_x, augmented_y, 1)# is_test=1
                output_class_t = output.argmax(dim=1) + 1  # output属于第几类

                loss_test = criterion(output, z.long() - 1)

                test_loss += loss_test.cpu().item()
                test_acc += (output_class_t == z).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1

                output_class_total.extend(np.array(output_class_t.cpu()))

        gt_test = test_gt.flatten()
        overall_acc = metrics.accuracy_score(gt_test, output_class_total)
        kappa = metrics.cohen_kappa_score(gt_test, output_class_total)
        confusion_matrix = metrics.confusion_matrix(gt_test, output_class_total)
        each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
        formatted_ea = ["%.4f" % ea for ea in each_acc]
        print("each_acc", formatted_ea)
        print("OA", "%.4f"%overall_acc)
        print("AA", "%.4f"%average_acc)
        print("Kappa", "%.4f"%kappa)

        # 写入txt文件
        f = open("./result_file_patch.txt", "a", encoding="UTF-8")
        # f.truncate(0)# 删除txt里内容
        f.write("\nthe dataset is {} seed_num: {} kappa: {} average_accurate: {} over_accurate: {}".format(dataset_index, seed,"%.4f"%kappa, "%.4f"%average_acc, "%.4f"%overall_acc))
        f.close()
        print("result is saved")
        print("max_point_acc:", max_point_acc)

        # # test edge patch
        # single_band = hsi_data[:,:,0]
        # sobel_edge = sobel(single_band)
        # sobel_edge_normalized = (sobel_edge - sobel_edge.min()) / (sobel_edge.max() - sobel_edge.min())
        # # 设定阈值，将其转换为 0-1 矩阵
        # threshold = 0.3
        # binary_edge = (sobel_edge_normalized > threshold).astype(np.uint8)
        # # 显示第一个波段的原始图像和 Sobel 滤波结果
        # plt.figure(figsize=(12, 6))
        # plt.subplot(121)
        # plt.imshow(binary_edge, cmap='gray')
        # plt.title('Original Band 0')
        # plt.axis('off')
        #
        # plt.subplot(122)
        # plt.imshow(sobel_edge, cmap='gray')
        # plt.title('Sobel Filtered Band 0')
        # plt.axis('off')
        #
        # plt.show()


        print('Start generating the entire classification map')
        net.eval()
        y_pred = []
        output_class_total = []
        with torch.no_grad():
            for x, y, _ in all_iter:
                x = x.to(device)
                y = y.to(device)

                _, output, _ = net(x, y)

                output_class_t = output.argmax(dim=1) + 1  # output属于第几类
                output_class_total.extend(np.array(output_class_t.cpu()))

                y_pred.extend(output.cpu().argmax(dim=1))

        h, w = gt_.shape
        y_pred = np.array(y_pred)
        pre_t = y_pred
        pre_t = pre_t.reshape(h, w) + 1

        # 示例分类数据，假设3x3的分类结果
        classification_map = pre_t
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

        # # 计算 gt_hsi * binary_edge
        edge_gt_mask = gt_hsi*binary_edge # 真正边缘
        pre_t[gt_hsi==0]=0
        result = pre_t*edge_gt_mask
        sss = result != edge_gt_mask

        # 创建一个自定义的颜色列表
        cmap_list = [color_dict[i] for i in sorted(color_dict.keys())]
        # 创建自定义的colormap
        cmap = mcolors.ListedColormap(cmap_list)
        # 定义颜色边界
        bounds = np.arange(len(color_dict) + 1) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        # 显示分类地图
        plt.imshow(sss, cmap='gray')
        plt.title('Hyperspectral Image Classification Map')
        plt.axis('off')

        output_path = './' + str(dataset_index) + '122925.png'  # 指定保存的路径和文件名
        # plt.savefig(output_path, dpi=800, bbox_inches='tight')
        # plt.show()

        plt.imshow(classification_map, cmap=cmap, norm=norm)
        plt.title('Hyperspectral Image Edge Map')
        plt.axis('off')
        output_path = './' + str(dataset_index) + '155925.png'  # 指定保存的路径和文件名
        plt.savefig(output_path, dpi=800, bbox_inches='tight')
        plt.show()


        # from thop import profile
        # print('==> Building model..')
        # model = net
        # input1 = torch.randn(1, 63, 9, 9).to(device)
        # input2 = torch.randn(1, 1, 9, 9).to(device)
        # flops, params = profile(model, (input1, input2))
        # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e3))

        print('over')



if __name__ == '__main__':
    print("------------------------------------")
    print("start saving the results")
    print("------------------------------------")


    main_functions(num = 1, ep = 200)

    print("all is finished")
