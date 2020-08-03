from torch.utils.data import DataLoader


class Data:

    def load_datasets(self):  # Loads self.dataset_{train,val,test}
        raise NotImplementedError

    def custom_collate_fn(self):
        raise NotImplementedError

    def get_loaders(self, batch_size, shuffle_train=False, num_workers=0,
                    get_test=True):
        try:
            collate_fn = self.custom_collate_fn
        except NotImplementedError:
            collate_fn = None

        try:
            loader_train = DataLoader(self.dataset_train,
                                      batch_size=batch_size,
                                      shuffle=shuffle_train,
                                      num_workers=num_workers,
                                      collate_fn=collate_fn)
            loader_val = DataLoader(self.dataset_val,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    collate_fn=collate_fn)
            loader_test = DataLoader(self.dataset_test,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn) \
                                     if get_test else None
        except AttributeError:
            print('Call load_datasets before getting loaders.')

        return loader_train, loader_val, loader_test
