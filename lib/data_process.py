import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset

def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)
    day_init = day_start
    week_init = week_start
    holiday_init = 1

    for index in range(day_start//interval, data.shape[0]+day_start//interval):
        if (index) % time_slot == 0 and index!=0:
            day_init = 0
        day_init = day_init + interval
        if (index) % time_slot == 0 and index !=0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1
        if day_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is None:
        k = 1
    else:
        for j in holiday_list :
            holiday_data[j-1 * time_slot:j * time_slot, :] = 2
    return day_data, week_data, holiday_data

# load dataset
def load_st_dataset(dataset, args):
    # =========== Traffic flow (PEMS) =========== #
    # 1 / 1 / 2018 - 2 / 28 / 2018 Monday
    if dataset == 'PEMS04':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :, 0]
        print(data.shape, data[data==0].shape)
        week_start = 1
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        holiday_list = [1, 15, 50]
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    # 7 / 1 / 2016 - 8 / 31 / 2016 Friday
    elif dataset == 'PEMS08':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :, 0]
        print(data.shape, data[data==0].shape)
        week_start = 5
        holiday_list = [4]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    #   9/1/2018 - 11/30/2018 Saturday
    elif dataset == 'PEMS03':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :, 0]
        week_start = 6
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    # 5 / 1 / 2017 - 8 / 31 / 2017 Monday
    elif dataset == 'PEMS07':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :, 0]
        week_start = 1
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # =========== Traffic index (DIDI) =========== #
    # 1 / 1 / 2018 - 4 / 30 / 2018 Monday
    elif dataset == 'CD_DIDI':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :, 0]
        week_start = 1
        holiday_list = [4]
        interval = 10
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    # 1 / 1 / 2018 - 4 / 30 / 2018 Monday
    elif dataset == 'SZ_DIDI':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :, 0]
        week_start = 1
        holiday_list = [4]
        interval = 10
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # =========== Traffic speed (PEMS) =========== #
    # 2012.03.01 - 2012.06.30
    elif dataset == 'METR_LA':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:(31+30)*288, :]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        week_start = 4
        holiday_list = None
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    # 2017.01.01 - 2017.06.30
    elif dataset == 'PEMS_BAY':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        if args.mode == 'pretrain':
            # 2017.01.01-2017.02.28 data
            data = np.load(data_path)['data'][:(31+28)*288, :]
        else:
            # 2017.01.01-2017.04.30 data
            data = np.load(data_path)['data'][:(31+28+31+30)*288, :]
        interval = 5
        week_day = 7
        args.interval = interval
        args.week_day = week_day
        week_start = 7
        holiday_list = None
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    # 5 / 1 / 2012 - 6 / 30 / 2012 Tuesday
    elif dataset == 'PEMS07M':
        data_path = os.path.join('../data/PEMS07M/PEMS07M.npz')
        data = np.load(data_path)['data']
        week_start = 2
        weekday_only = True
        interval = 5
        week_day = 5
        args.interval = interval
        args.week_day = week_day
        holiday_list = []
        day_data, week_data, holiday_data = time_add(data, week_start, interval, weekday_only, holiday_list=holiday_list)

    # =========== Taxi demand (NYC CHI Opendata) =========== #
    # 1 / 1 / 2016 - 12 / 31 / 2021 Friday
    elif dataset == 'NYC_TAXI':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        if args.mode == 'pretrain':
            # 2016-2020 data
            data = np.load(data_path)['data'][:, :int((366+365+365+365+366)*48), 0].transpose(1, 0)
        else:
            data = np.load(data_path)['data'][..., 0].transpose(1, 0)
        data[np.isnan(data)] = 0
        week_start = 5
        interval = 30
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
    # 1 / 1 / 2021 - 12 / 31 / 2021 Friday
    elif dataset == 'CHI_TAXI':
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :, 0].transpose(1, 0)
        data[np.isnan(data)] = 0
        week_start = 5
        interval = 30
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # =========== Bike trajectory (NYC CHI Opendata) =========== #
    # 1 / 1 / 2016 - 12 / 31 / 2021 Friday
    elif dataset.startswith("NYC_BIKE"):
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        # only 2020 data
        data = np.load(data_path)['data'][(366+365+365+365)*48:(366+365+365+365+366)*48, :, 0]
        data[np.isnan(data)] = 0
        week_start = 3
        interval = 30
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    # =========== Traffic speed (UniST) =========== #
    # 3 / 7 / 2022 - 4 / 5 / 2022 Friday
    elif dataset.startswith("Traffic"):
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][:, :]
        data[np.isnan(data)] = 0
        week_start = 1
        interval = 30
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False,
                                                     holiday_list=holiday_list, day_start=36*30)

    # =========== Traffic flow (LargeST) =========== #
    # 1 / 1 / 2020 - 12 / 31 / 2020
    elif dataset.startswith("CAD") and "-" in dataset:
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        if args.mode == 'pretrain':
            data = np.load(data_path)['data'][:288*(31+29), :]
        else:
            data = np.load(data_path)['data'][:288 * (31+29+15), :]
        data[np.isnan(data)] = 0
        week_start = 3
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)

    else:
        data_path = os.path.join(f'../data/{dataset}/{dataset}.npz')
        data = np.load(data_path)['data'][288*(31+29):288*(31+29+31+30), :]
        data[np.isnan(data)] = 0
        week_start = 3
        interval = 5
        week_day = 7
        holiday_list = None
        args.interval = interval
        args.week_day = week_day
        day_data, week_data, holiday_data = time_add(data, week_start, interval=interval, weekday_only=False, holiday_list=holiday_list)
        # raise ValueError

    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
        day_data = np.expand_dims(day_data, axis=-1).astype(int)
        week_data = np.expand_dims(week_data, axis=-1).astype(int)
        # holiday_data = np.expand_dims(holiday_data, axis=-1).astype(int)
        data = np.concatenate([data, day_data, week_data], axis=-1)
    elif len(data.shape) > 2:
        print(args.data_type)
        if args.data_type == 'crime':
            week_data = np.expand_dims(week_data, axis=-1).astype(int)
            data = np.concatenate([data, week_data], axis=-1)
        elif args.data_type == 'demand' or args.data_type == 'people flow':
            day_data = np.expand_dims(day_data, axis=-1).astype(int)
            week_data = np.expand_dims(week_data, axis=-1).astype(int)
            data = np.concatenate([data, day_data, week_data], axis=-1)
        else:
            raise ValueError

    print('Load %s Dataset shaped: ' % dataset, data.shape, data[..., 0:1].max(), data[..., 0:1].min(),
          data[..., 0:1].mean(), np.median(data[..., 0:1]), data.dtype)
    return data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    if test_ratio == 0:
        test_data = data[:0]
        val_data = data[-int(data_len * (test_ratio + val_ratio)):-int(data_len * test_ratio)]
        if val_ratio == 0:
            train_data = data
        else:
            train_data = data[:-int(data_len * (test_ratio + val_ratio))]
    else:
        test_data = data[-int(data_len*test_ratio):]
        val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
        train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean

def normalize_dataset(data, input_base_dim):
    data_ori = data[:, :, 0:input_base_dim]
    data_day = data[:, :, input_base_dim:input_base_dim+1]
    data_week = data[:, :, input_base_dim+1:input_base_dim+2]

    mean_data = data_ori.mean()
    std_data = data_ori.std()
    mean_day = data_day.mean()
    std_day = data_day.std()
    mean_week = data_week.mean()
    std_week = data_week.std()

    scaler_data = StandardScaler(mean_data, std_data)
    scaler_day = StandardScaler(mean_day, std_day)
    scaler_week = StandardScaler(mean_week, std_week)
    print('Normalize the dataset by Standard Normalization')

    return scaler_data, scaler_day, scaler_week


class TrafficDataset(Dataset):
    def __init__(self, data, batch_size, input_window=288, output_window=288, eval_only=False):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window

        # preprocess
        self.windows = [
            (data[i:i + input_window], data[i + input_window:i + input_window + output_window])
            for i in range(len(data) - input_window - output_window + 1)
        ]

        # drop_last & shuffle
        if eval_only==False:
            random.shuffle(self.windows)
            if len(self.windows) % batch_size != 0:
                self.windows = self.windows[:-(len(self.windows) % batch_size)]


        # batch
        self.batches = [
            self.windows[i:i + batch_size]
            for i in range(0, len(self.windows), batch_size)
        ]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        batch_x, batch_y = zip(*self.batches[idx])
        return torch.from_numpy(np.stack(batch_x)).float(), torch.from_numpy(np.stack(batch_y)).float()

def define_dataloder(args):
    dataloder_trn_list, dataloder_val_list, dataloder_tst_list = [], [], []
    scaler_dict = {}
    num_nodes_dict = {}

    # node_nums = 0
    # tp_nums = 0
    # for dataset_name in args.dataset_use:
    #     print(args.dataset_use, dataset_name, args.val_ratio, args.test_ratio)
    #     # print(sss)
    #     data = load_st_dataset(dataset_name, args)
    #     node_nums = node_nums + data.shape[1]
    #     tp_nums = tp_nums + data.shape[0]
    #     print(dataset_name, data.shape)
    #     # num_nodes_dict[dataset_name] = data.shape[1]
    #     # data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
    #     # print('data_train', data_train.shape, data_val.shape, data_test.shape)
    #     # if args.real_value == False:
    #     #     scaler_data, scaler_day, scaler_week = normalize_dataset(data_train, args.input_base_dim)
    #     #     print(data_train.shape, scaler_data.mean, scaler_data.std)
    #     #     data_train[..., :args.input_base_dim] = scaler_data.transform(data_train[:, :, :args.input_base_dim])
    #     #     data_val[..., :args.input_base_dim] = scaler_data.transform(data_val[:, :, :args.input_base_dim])
    #     #     data_test[..., :args.input_base_dim] = scaler_data.transform(data_test[:, :, :args.input_base_dim])
    #     #     scaler_dict[dataset_name] = scaler_data
    #     # else:
    #     #     scaler_dict[dataset_name] = None
    #     # if dataset_name.startswith("Traffic") or dataset_name.startswith("NYC") or dataset_name.startswith("CHI"):
    #     #     intervel = 30
    #     # else:
    #     #     intervel = 5
    #     # iw = args.his // (intervel // 5)
    #     # ow = args.pred // (intervel // 5)
    #     # datasets_train = TrafficDataset(data_train, args.batch_size, input_window=iw, output_window=ow, eval_only=False)
    #     # datasets_val = TrafficDataset(data_val, args.batch_size, input_window=iw, output_window=ow, eval_only=True)
    #     # datasets_test = TrafficDataset(data_test, args.batch_size, input_window=iw, output_window=ow, eval_only=True)
    #     # dataloder_trn_list.append(datasets_train)
    #     # dataloder_val_list.append(datasets_val)
    #     # dataloder_tst_list.append(datasets_test)
    # print(tp_nums, node_nums)
    # print(sss)

    for dataset_name in args.dataset_use:
        print(args.dataset_use, dataset_name, args.val_ratio, args.test_ratio)
        # print(sss)
        data = load_st_dataset(dataset_name, args)
        num_nodes_dict[dataset_name] = data.shape[1]
        data_train, data_val, data_test = split_data_by_ratio(data, args.val_ratio, args.test_ratio)
        print('data_train', data_train.shape, data_val.shape, data_test.shape)
        if args.real_value == False:
            scaler_data, scaler_day, scaler_week = normalize_dataset(data_train, args.input_base_dim)
            print(data_train.shape, scaler_data.mean, scaler_data.std)
            data_train[..., :args.input_base_dim] = scaler_data.transform(data_train[:, :, :args.input_base_dim])
            data_val[..., :args.input_base_dim] = scaler_data.transform(data_val[:, :, :args.input_base_dim])
            data_test[..., :args.input_base_dim] = scaler_data.transform(data_test[:, :, :args.input_base_dim])
            scaler_dict[dataset_name] = scaler_data
        else:
            scaler_dict[dataset_name] = None
        if dataset_name.startswith("Traffic") or dataset_name.startswith("NYC") or dataset_name.startswith("CHI"):
            intervel = 30
        elif 'DIDI' in dataset_name:
            intervel = 10
        else:
            intervel = 5
        iw = args.his // (intervel // 5)
        ow = args.pred // (intervel // 5)
        datasets_train = TrafficDataset(data_train, args.batch_size, input_window=iw, output_window=ow, eval_only=False)
        datasets_val = TrafficDataset(data_val, args.batch_size, input_window=iw, output_window=ow, eval_only=True)
        datasets_test = TrafficDataset(data_test, args.batch_size, input_window=iw, output_window=ow, eval_only=True)
        dataloder_trn_list.append(datasets_train)
        dataloder_val_list.append(datasets_val)
        dataloder_tst_list.append(datasets_test)

    train_combine, val_combine, test_combine = ConcatDataset(dataloder_trn_list), ConcatDataset(dataloder_val_list), ConcatDataset(dataloder_tst_list)

    # train_dataloader = DataLoader(train_combine, batch_size=1, shuffle=True)
    # val_dataloader = DataLoader(val_combine, batch_size=1, shuffle=False)
    # test_dataloader = DataLoader(test_combine, batch_size=1, shuffle=False)
    if (1-args.val_ratio-args.test_ratio) <= 0:
        train_dataloader = None
    else:
        train_dataloader = DataLoader(train_combine, batch_size=1, shuffle=True)
    if args.val_ratio <= 0:
        val_dataloader = None
    else:
        val_dataloader = DataLoader(val_combine, batch_size=1, shuffle=False)
    if args.test_ratio <= 0:
        test_dataloader = 0
    else:
        test_dataloader = DataLoader(test_combine, batch_size=1, shuffle=False)
    args.num_nodes_dict = num_nodes_dict
    return train_dataloader, val_dataloader, test_dataloader, scaler_dict

def get_key_from_value(d, value):
    for key, val in d.items():
        if val == value:
            return key
    return None

