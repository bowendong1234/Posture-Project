from keras.utils import Sequence
import numpy as np

class CombinedDataGenerator(Sequence):
    def __init__(self, generator1, generator2):
        self.generator1 = generator1
        self.generator2 = generator2
        self.batch_size = self.generator1.batch_size
        assert len(generator1) == len(generator2), "Generators must have the same length."

    def __len__(self):
        return len(self.generator1)

    def __getitem__(self, index):
        X1_batch, y1_batch = self.generator1[index]
        X2_batch, y2_batch = self.generator2[index]
        # Concatenate the batches
        X_batch = np.concatenate((X1_batch, X2_batch), axis=0)
        y_batch = np.concatenate((y1_batch, y2_batch), axis=0)
        return X_batch, y_batch

