
import numpy as np
import pandas as pd
import os

def CAD_data_generate(is_split, select_block, region_select=-10, region4_1=None, region4_2=None, region4_3=None, region4_4=None):
    ca_meta = pd.read_csv('./ca_meta.csv')
    sd_meta = ca_meta[ca_meta.District == select_block]
    sd_meta = sd_meta.reset_index()
    sd_meta = sd_meta.drop(columns=['index'])

    if is_split:
        dataset_name = f'CAD{select_block}-{region_select}'
        if region_select == 1:
            start_index = 0
            end_index = region4_1
        elif region_select == 2:
            start_index = region4_1
            end_index = region4_1 + region4_2
        elif region_select == 3:
            start_index = region4_1 + region4_2
            end_index = region4_1 + region4_2 + region4_3
        elif region_select == 4:
            start_index = region4_1 + region4_2 + region4_3
            end_index = region4_1 + region4_2 + region4_3 + region4_4
        print(f'CAD{select_block}-{region_select}', 'is started!!!')
    else:
        dataset_name = f'CAD{select_block}'
        start_index = 0
        end_index = None
        print(f'CAD{select_block}', 'is started!!!')



    directory = f'./{dataset_name}'
    if not os.path.exists(directory):
        os.mkdir(directory)

    print(sd_meta[start_index:end_index])
    sd_meta.to_csv(f'./{dataset_name}/{dataset_name}_meta.csv', index=False)
    print(sd_meta[sd_meta.duplicated(subset=['Lat', 'Lng'])])

    sd_meta_id2 = sd_meta.ID2.values.tolist()
    print(len(sd_meta_id2))
    # print(sss)

    ca_rn_adj = np.load('./ca_rn_adj.npy')
    print(ca_rn_adj.shape)

    sd_rn_adj = ca_rn_adj[sd_meta_id2]
    sd_rn_adj = sd_rn_adj[:,sd_meta_id2]
    sd_rn_adj_select = sd_rn_adj[start_index:end_index, start_index:end_index]
    print(sd_rn_adj_select.shape)

    np.save(f'./{dataset_name}/{dataset_name}_rn_adj.npy', sd_rn_adj_select)

    year = '2020'  # please specify the year, our experiments use 2019

    sd_meta.ID = sd_meta.ID.astype(str)
    sd_meta_id = sd_meta.ID.values.tolist()

    ca_his = pd.read_hdf('./ca_his_raw_' + year +'.h5')
    sd_his = ca_his[sd_meta_id]
    print('sd_his', sd_his.shape)
    sd_his_select = sd_his.values[..., start_index:end_index]
    print(sd_his_select)
    np.savez(f'./{dataset_name}/{dataset_name}.npz', data=sd_his_select)
    sd_his_npz = np.load(f'./{dataset_name}/{dataset_name}.npz')['data']
    print(sd_his_npz.shape)

    # code for checking adj stat
    sd_rn_adj = np.load(f'./{dataset_name}/{dataset_name}_rn_adj.npy')
    node_num = sd_rn_adj.shape[0]

    print(sd_rn_adj[0,0])
    sd_rn_adj[np.arange(node_num), np.arange(node_num)] = 0
    print(sd_rn_adj[0,0])

    print('edge number', np.count_nonzero(sd_rn_adj))
    print('node degree', np.mean(np.count_nonzero(sd_rn_adj, axis=-1)))
    print('sparsity', np.count_nonzero(sd_rn_adj) / (node_num**2) * 100)

select_block_list = [3, 4, 5, 7, 8, 12]

for select_block in select_block_list:
    if select_block == 3 or select_block == 5:
        is_split = False
        CAD_data_generate(is_split, select_block)
    else:
        is_split = True
        # 4
        if select_block == 4:
            region_select_list = [1, 2, 3, 4]
            region4_1 = 621
            region4_2 = 610
            region4_3 = 593
            region4_4 = 528
        # 7
        elif select_block == 7:
            region_select_list = [1, 2, 3]
            region4_1 = 666
            region4_2 = 634
            region4_3 = 559
            region4_4 = None
        # 8
        elif select_block == 8:
            region_select_list = [1, 2]
            region4_1 = 510
            region4_2 = 512
            region4_3 = None
            region4_4 = None
        # 12
        elif select_block == 12:
            region_select_list = [1, 2]
            region4_1 = 453
            region4_2 = 500
            region4_3 = None
            region4_4 = None

        for region_select in region_select_list:
            CAD_data_generate(is_split, select_block, region_select,
                              region4_1, region4_2, region4_3, region4_4)





