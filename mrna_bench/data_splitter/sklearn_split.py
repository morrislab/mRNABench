from sklearn.model_selection import train_test_split

from mrna_bench.data_splitter.data_splitter import DataSplitter


class SklearnSplitter(DataSplitter):
    def split_df(self, df, test_size, random_seed):
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed
        )
