import numpy as np
from operator import truediv
import torch
import torch.utils.data as Data
from Utils.extract_samll_cubic import select_small_cubic


def patch_handle(data, lidar_data, train_id, test_id, gt, patch_length, batch_size, dataset_index):
    data_x, data_y, band, lidar_band = data.shape[0], data.shape[1], data.shape[2], lidar_data.shape[2]
    # 切片处理
    patch_l = patch_length
    patch_size = 2 * patch_length + 1 #9*9
    input_dimension = band

    gt_reshape = gt.reshape(-1)
    train_size = len(train_id)
    test_size = len(test_id)
    total_size = train_size + test_size
    all_size = len(gt_reshape)
    train_indices = train_id
    test_indices = test_id
    total_indices = train_indices + test_indices
    all_indices = [index for index, sequence in enumerate(gt_reshape)]

    whole_data = data.reshape((data_x, data_y, band))
    whole_lidar_data = lidar_data.reshape((data_x, data_y, lidar_band))
    padded_data = np.lib.pad(whole_data, ((patch_l, patch_l), (patch_l, patch_l), (0, 0)), 'constant', constant_values=0)
    padded_lidar_data = np.lib.pad(whole_lidar_data, ((patch_l, patch_l), (patch_l, patch_l), (0, 0)), 'constant', constant_values=0)

    # hsi数据
    train_data = select_small_cubic(train_size, train_indices, whole_data, patch_l, padded_data, input_dimension)  # [160, 9, 9, 200]
    test_data = select_small_cubic(test_size, test_indices, whole_data, patch_l, padded_data, input_dimension)
    all_data = select_small_cubic(all_size, all_indices, whole_data, patch_l, padded_data, input_dimension)

    # lidar数据
    train_lidar_data = select_small_cubic(train_size, train_indices, whole_lidar_data, patch_l, padded_lidar_data, lidar_band)  # [160, 9, 9, 2]
    test_lidar_data = select_small_cubic(test_size, test_indices, whole_lidar_data, patch_l, padded_lidar_data, lidar_band)
    all_lidar_data = select_small_cubic(all_size, all_indices, whole_lidar_data, patch_l, padded_lidar_data, lidar_band)

    gt_hsi_ = gt.reshape(-1)
    gt_train = gt_hsi_[train_indices]
    gt_test = gt_hsi_[test_indices]
    gt_total = gt_hsi_[total_indices]
    gt_all = gt_hsi_[all_indices]

    tensor_train_data = torch.from_numpy(train_data).permute(0, 3, 1, 2).type(torch.FloatTensor)
    tensor_train_lidar_data = torch.from_numpy(train_lidar_data).permute(0, 3, 1, 2).type(torch.FloatTensor)
    tensor_train_gt = torch.from_numpy(gt_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(tensor_train_data, tensor_train_lidar_data, tensor_train_gt)

    tensor_test_data = torch.from_numpy(test_data).permute(0, 3, 1, 2).type(torch.FloatTensor)
    tensor_test_lidar_data = torch.from_numpy(test_lidar_data).permute(0, 3, 1, 2).type(torch.FloatTensor)
    tensor_test_gt = torch.from_numpy(gt_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(tensor_test_data, tensor_test_lidar_data, tensor_test_gt)

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], input_dimension)
    all_tensor_data = torch.from_numpy(all_data).permute(0, 3, 1, 2).type(torch.FloatTensor)
    all_tensor_lidar_data = torch.from_numpy(all_lidar_data).permute(0, 3, 1, 2).type(torch.FloatTensor)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_lidar_data, all_tensor_data_label)



    batch_size = batch_size
    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,
        num_workers=0,
    )

    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,
        num_workers=0,
    )

    # all_iter = 0
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,
        num_workers=0
    )

    return train_iter, test_iter, all_iter


def create_pairs(contrastive_pairs, contrastive_feature, class_num):
    contrastive_feature = contrastive_feature.detach().numpy()
    b, c, h, w = contrastive_feature.shape[0], contrastive_feature.shape[1], contrastive_feature.shape[2], \
    contrastive_feature.shape[3]
    contrastive_feature = contrastive_feature.reshape(b, c, h * w)
    query_index = contrastive_pairs[0::2]
    positive_index = contrastive_pairs[1::2]
    negative_index = []
    query = []
    positive = []
    negative = []
    for i in range(class_num):
        query.append(contrastive_feature[0,:,query_index[i]])
        positive.append(contrastive_feature[0,:,positive_index[i]])
    for i in range(class_num):
        tem_list = query + positive
        del tem_list[i:i+1]
        del tem_list[i+class_num-1:i+class_num]
        negative.append(tem_list)
    query = torch.tensor(np.array(query))
    positive = torch.tensor(np.array(positive))
    negative = torch.tensor(np.array(negative))

    return query, positive, negative

# test loss
# loss = InfoNCE(negative_mode='paired')
# batch_size, num_negative, embedding_size = 11, 6, 128
# query = torch.randn(batch_size, embedding_size)#(11, 128)
# positive_key = torch.randn(batch_size, embedding_size)
# negative_keys = torch.randn(batch_size, num_negative, embedding_size)
# output = loss(query, positive_key, negative_keys)
# print(output)
#
# contrastive_pairs = torch.arange(0, 22)
# contrastive_feature = torch.randn(1, 128, 336, 224)
# class_num = 11
#
# query, positive, negative = create_pairs(contrastive_pairs, contrastive_feature, 11)
# output = loss(query, positive, negative)
# print(output)


def segment_data(hsi_data, lidar_data, gt_hsi):
    # segmentate dataset
    seg_hsi_data = []
    seg_lidar_data = []
    seg_gt_hsi = []
    height_, width_, channel_ = hsi_data.shape
    middle_height = height_ // 2
    middle_width = width_ // 2
    for i in range(4):  # 分成四份
        if i == 0:
            image_seg = hsi_data[:middle_height, :middle_width, :]
            lidar_seg = lidar_data[:middle_height, :middle_width, :]
            gt_seg = gt_hsi[:middle_height, :middle_width]
        elif i == 1:
            image_seg = hsi_data[middle_height:, :middle_width, :]
            lidar_seg = lidar_data[middle_height:, :middle_width, :]
            gt_seg = gt_hsi[middle_height:, :middle_width]
        elif i == 2:
            image_seg = hsi_data[:middle_height, middle_width:, :]
            lidar_seg = lidar_data[:middle_height, middle_width:, :]
            gt_seg = gt_hsi[:middle_height, middle_width:]
        elif i == 3:
            image_seg = hsi_data[middle_height:, middle_width:, :]
            lidar_seg = lidar_data[middle_height:, middle_width:, :]
            gt_seg = gt_hsi[middle_height:, middle_width:]
        seg_hsi_data.append(image_seg)
        seg_lidar_data.append(lidar_seg)
        seg_gt_hsi.append(gt_seg)
    seg_hsi_data = np.array(seg_hsi_data)
    seg_lidar_data = np.array(seg_lidar_data)
    seg_gt_hsi = np.array(seg_gt_hsi)

    return seg_hsi_data, seg_lidar_data, seg_gt_hsi

def segment_data_2(gt_hsi):
    # segmentate dataset
    seg_hsi_data = []
    seg_lidar_data = []
    seg_gt_hsi = []
    height_, width_ = gt_hsi.shape
    middle_height = height_ // 2
    middle_width = width_ // 2
    for i in range(4):  # 分成四份
        if i == 0:
            gt_seg = gt_hsi[:middle_height, :middle_width]
        elif i == 1:
            gt_seg = gt_hsi[middle_height:, :middle_width]
        elif i == 2:
            gt_seg = gt_hsi[:middle_height, middle_width:]
        elif i == 3:
            gt_seg = gt_hsi[middle_height:, middle_width:]
        seg_gt_hsi.append(gt_seg)
    seg_gt_hsi = np.array(seg_gt_hsi)
    return seg_gt_hsi

def sampling_num(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    class_sum = []
    select_num = []
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        class_sum.append(len(indexes))
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if len(indexes) <= 40:
            nb_val = int(len(indexes) / 2)
        else:
            nb_val = proportion
        select_num.append(nb_val)
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    contrastive_pairs = []
    for i in range(m):
        train_indexes += train[i]
        contrastive_pairs += train[i][0:2]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes, class_sum, select_num

def sampling_pro(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    class_sum = []
    select_num = []
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        class_sum.append(len(indexes))
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        # if proportion != 1:
        #     nb_val = max(int((1 - proportion) * len(indexes)), 3)
        # else:
        #     nb_val = 0

        if proportion == 1:
            nb_val = 0
        elif len(indexes) <= 500:
            nb_val = 40
        else:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)

        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        select_num.append(nb_val)
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes, class_sum, select_num


def random_mask(h, w):# 尺寸
    # 定义掩码尺寸
    mask_size = (h, w)
    # 生成随机张量
    random_mask = torch.rand(mask_size)
    # 设定阈值，将随机张量转换为布尔掩码张量
    threshold = 0.5
    random_mask = random_mask < threshold
    # 打印结果
    return random_mask

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    print(list_diag)
    print(list_raw_sum)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

