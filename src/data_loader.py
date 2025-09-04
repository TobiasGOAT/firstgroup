import pickle
import os
class DataLoader:
    def __init__(self):
        self.train, self.valid, self.test = self.load_data()

    def load_data(self):
        """Load MNIST train, validation, and test data."""
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'mnist.pkl')
        with open(data_path, 'rb') as f:
            train, valid, test = pickle.load(f, encoding='latin-1')
        return train, valid, test
    
    def get_data_information(self):
        """Return the shape of the datasets, their data types, unique labels and data range."""
        return {
            'train': (self.train[0].shape, self.train[1].shape),
            'train_data_type': type(self.train[0]),
            'valid': (self.valid[0].shape, self.valid[1].shape),
            'valid_data_type': type(self.valid[0]),
            'test': (self.test[0].shape, self.test[1].shape),
            'test_data_type': type(self.test[0]),
            'unique_labels': set(self.train[1]),
            'training_data_range': (self.train[0].min(), self.train[0].max())
        }
    
    def get_train_data(self):
        """Return training data and labels."""
        return self.train

    def get_valid_data(self):
        """Return validation data and labels."""
        return self.valid

    def get_test_data(self):
        """Return test data and labels."""
        return self.test
    
    def get_all_data(self):
        """Return all datasets."""
        return self.train, self.valid, self.test
    
    def print_data_summary(self):
        """Print a summary of the datasets."""
        info = self.get_data_information()
        print("Data Summary:")
        print(f"Training set: {info['train'][0]} samples, Labels: {info['train'][1]}")
        print(f"Validation set: {info['valid'][0]} samples, Labels: {info['valid'][1]}")
        print(f"Test set: {info['test'][0]} samples, Labels: {info['test'][1]}")
        print(f"Unique labels in training set: {info['unique_labels']}")
        print(f"Training data range: {info['data_range']}")

if __name__ == "__main__":
    loader = DataLoader()
    loader.print_data_summary()
