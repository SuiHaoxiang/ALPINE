from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.ucr_loader import UCRDataset
from data_provider.ucr_csv_loader import UCRCSVDataset
from data_provider.sd_loader import SD_Loader
from data_provider.fd_loader import FD_Loader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
import torch
from models.ModernTCN import Model
from models.ModernTCN_3D import ModernTCN_3D
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'UCR': UCRDataset,
    'UCR_CSV': UCRCSVDataset,
    'SD': SD_Loader,
    'FD': FD_Loader
}


def select_model(args, configs):
    # Ensure passing actual input dimensions
    if hasattr(args, 'actual_enc_in'):
        configs.actual_enc_in = args.actual_enc_in
        configs.enc_in = args.actual_enc_in
    
    # Select model based on model parameter
    if hasattr(args, 'model'):
        if args.model == 'ModernTCN_3D':
            return ModernTCN_3D(configs).float()
        elif args.model == 'ModernTCN':
            return Model(configs).float()
    
    # Default behavior: use 3D model for SD/FD data, normal model for others
    if 'SD' in args.data or 'FD' in args.data:
        return ModernTCN_3D(configs).float()
    return Model(configs).float()

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        print(f"Loading {flag} data from: {args.root_path}")
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        if hasattr(data_set, '__len__'):
            print(f"{flag} dataset loaded - samples: {len(data_set)}, first sample shape: {data_set[0][0].shape if len(data_set)>0 else 'empty'}")
        else:
            print(f"{flag} dataset loaded - data provider doesn't support length query")
        def custom_collate(batch):
            # Handle different cases for train/test sets
            x = torch.stack([item[0] for item in batch])
            y = None
            if batch[0][1] is not None:  # If labels exist
                y = torch.stack([item[1] for item in batch])
            # Remove redundant dimensions (batch,1,100,1) -> (batch,100,1)
            if len(x.shape) == 4:  # For UCR data format
                x = x.squeeze(1)
            return x, y

        if hasattr(data_set, '__len__'):
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=custom_collate)
        else:
            # For data loaders without __len__ implementation, use fixed batch count
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,  # Cannot shuffle without length information
                num_workers=args.num_workers,
                drop_last=False,
                collate_fn=custom_collate)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
