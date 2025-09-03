import pickle
import os
import numpy as np
from typing import Tuple, Optional


def load_mnist_data(data_path: Optional[str] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load MNIST dataset from pickle file.
    
    Args:
        data_path (str, optional): Path to the pickle file. If None, uses default path.
        
    Returns:
        tuple: ((train_data, train_labels), (test_data, test_labels))
               where each array contains the respective dataset split.
    """
    if data_path is None:
        # Get the directory of this file and construct path to data folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, 'data', 'mnist.pkl')
    
    try:
        with open(data_path, 'rb') as f:
            # Try latin-1 encoding first (most common for older pickle files)
            data = pickle.load(f, encoding='latin-1')
        
        # Handle different possible pickle formats
        if isinstance(data, tuple) and len(data) == 2:
            # Format: (train_data, test_data) where each is (X, y)
            (train_X, train_y), (test_X, test_y) = data
        elif isinstance(data, tuple) and len(data) == 3:
            # Some MNIST formats have 3 elements: (train, valid, test)
            (train_X, train_y), (valid_X, valid_y), (test_X, test_y) = data
        elif isinstance(data, dict):
            # Format: dictionary with keys for train/test data
            # Handle both string and byte keys
            keys = list(data.keys())
            if b'train_data' in keys or 'train_data' in keys:
                train_key = b'train_data' if b'train_data' in keys else 'train_data'
                train_label_key = b'train_labels' if b'train_labels' in keys else 'train_labels'
                test_key = b'test_data' if b'test_data' in keys else 'test_data'
                test_label_key = b'test_labels' if b'test_labels' in keys else 'test_labels'
                
                train_X = data[train_key]
                train_y = data[train_label_key]
                test_X = data[test_key]
                test_y = data[test_label_key]
            else:
                # Try common alternative key names
                train_X = data.get('X_train', data.get(b'X_train'))
                train_y = data.get('y_train', data.get(b'y_train'))
                test_X = data.get('X_test', data.get(b'X_test'))
                test_y = data.get('y_test', data.get(b'y_test'))
                
                if train_X is None:
                    raise ValueError(f"Could not find training data. Available keys: {list(data.keys())}")
        else:
            raise ValueError(f"Unexpected pickle file format. Data type: {type(data)}, Length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
            
        return (train_X, train_y), (test_X, test_y)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find data file at: {data_path}")
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {str(e)}")


def load_data(data_path: Optional[str] = None, normalize: bool = True) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load and optionally preprocess MNIST data.
    
    Args:
        data_path (str, optional): Path to the pickle file. If None, uses default path.
        normalize (bool): Whether to normalize pixel values to [0, 1] range.
        
    Returns:
        tuple: ((train_data, train_labels), (test_data, test_labels))
    """
    (train_X, train_y), (test_X, test_y) = load_mnist_data(data_path)
    
    if normalize:
        # Normalize pixel values to [0, 1] range
        train_X = train_X.astype(np.float32) / 255.0
        test_X = test_X.astype(np.float32) / 255.0
    
    return (train_X, train_y), (test_X, test_y)


def debug_pickle_format(data_path: Optional[str] = None) -> None:
    """
    Debug function to inspect the pickle file format.
    
    Args:
        data_path (str, optional): Path to the pickle file.
    """
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        data_path = os.path.join(project_root, 'data', 'mnist.pkl')
    
    print(f"Debugging pickle file: {data_path}")
    
    try:
        with open(data_path, 'rb') as f:
            # Try different encoding methods
            encodings = ['latin-1', 'bytes', None]
            for encoding in encodings:
                try:
                    f.seek(0)
                    if encoding:
                        data = pickle.load(f, encoding=encoding)
                        print(f"\nSuccessfully loaded with encoding='{encoding}'")
                    else:
                        data = pickle.load(f)
                        print(f"\nSuccessfully loaded with default encoding")
                    
                    print(f"Data type: {type(data)}")
                    print(f"Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                    
                    if isinstance(data, tuple):
                        print(f"Tuple elements: {len(data)}")
                        for i, item in enumerate(data):
                            print(f"  Element {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
                    elif isinstance(data, dict):
                        print(f"Dictionary keys: {list(data.keys())}")
                        for key in data.keys():
                            value = data[key]
                            print(f"  {key}: type={type(value)}, shape={getattr(value, 'shape', 'N/A')}")
                    
                    break  # If successful, break out of the loop
                    
                except Exception as e:
                    print(f"Failed with encoding='{encoding}': {e}")
                    continue
                    
    except Exception as e:
        print(f"Could not open file: {e}")


def get_data_info(data_path: Optional[str] = None) -> dict:
    """
    Get information about the loaded dataset.
    
    Args:
        data_path (str, optional): Path to the pickle file.
        
    Returns:
        dict: Information about the dataset including shapes and data types.
    """
    (train_X, train_y), (test_X, test_y) = load_mnist_data(data_path)
    
    info = {
        'train_samples': train_X.shape[0],
        'test_samples': test_X.shape[0],
        'input_shape': train_X.shape[1:],
        'num_classes': len(np.unique(train_y)),
        'train_data_type': train_X.dtype,
        'train_labels_type': train_y.dtype,
        'data_range': (train_X.min(), train_X.max()),
        'unique_labels': sorted(np.unique(train_y).tolist())
    }
    
    return info


if __name__ == "__main__":
    # First debug the pickle format
    print("=== DEBUGGING PICKLE FORMAT ===")
    debug_pickle_format()
    
    print("\n=== LOADING DATA ===")
    # Example usage
    try:
        # Load the data
        (train_X, train_y), (test_X, test_y) = load_data()
        
        # Print dataset information
        info = get_data_info()
        print("Dataset Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
            
        print(f"\nSuccessfully loaded MNIST data!")
        print(f"Training set: {train_X.shape} samples")
        print(f"Test set: {test_X.shape} samples")
        
    except Exception as e:
        print(f"Error: {e}")