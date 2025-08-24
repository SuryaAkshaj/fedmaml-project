import unittest
import torch
import numpy as np
import tempfile
import os
from collections import OrderedDict

# Import the modules to test
from utils import (
    get_device, clone_model_state, apply_state, add_dp_noise,
    evaluate, average_states, plot_curves, split_train_val_from_loader
)
from model import make_model, ConvNetMNIST, ConvNetCIFAR10
from data_loader import set_seed, dirichlet_non_iid_splits

class TestUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = get_device(use_gpu=False)  # Use CPU for testing
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_get_device(self):
        """Test device selection"""
        # Test CPU selection
        device = get_device(use_gpu=False)
        self.assertEqual(device.type, 'cpu')
        
        # Test GPU selection (if available)
        if torch.cuda.is_available():
            device = get_device(use_gpu=True)
            self.assertEqual(device.type, 'cuda')
    
    def test_clone_model_state(self):
        """Test model state cloning"""
        model = ConvNetMNIST()
        state = clone_model_state(model)
        
        # Check that state is a copy, not a reference
        self.assertIsInstance(state, OrderedDict)
        self.assertEqual(len(state), len(model.state_dict()))
        
        # Check that modifying state doesn't affect model
        for key in state:
            state[key] = torch.randn_like(state[key])
        
        # Model should remain unchanged
        for key in model.state_dict():
            self.assertFalse(torch.equal(state[key], model.state_dict()[key]))
    
    def test_apply_state(self):
        """Test applying state to model"""
        model = ConvNetMNIST()
        original_state = clone_model_state(model)
        
        # Create new state with random values
        new_state = OrderedDict()
        for key, value in original_state.items():
            new_state[key] = torch.randn_like(value)
        
        # Apply new state
        apply_state(model, new_state)
        
        # Check that model state matches new state
        for key in new_state:
            self.assertTrue(torch.equal(new_state[key], model.state_dict()[key]))
    
    def test_add_dp_noise(self):
        """Test differential privacy noise addition"""
        # Create test state
        state = OrderedDict({
            'conv1.weight': torch.randn(32, 1, 3, 3),
            'conv1.bias': torch.randn(32),
            'fc1.weight': torch.randn(128, 64 * 7 * 7)
        })
        
        # Test without noise
        noiseless = add_dp_noise(state, std=0.0, device=self.device)
        for key in state:
            self.assertTrue(torch.equal(state[key], noiseless[key]))
        
        # Test with noise
        noisy = add_dp_noise(state, std=0.1, device=self.device)
        for key in state:
            self.assertFalse(torch.equal(state[key], noisy[key]))
            # Check that noise is reasonable
            diff = torch.abs(state[key] - noisy[key])
            self.assertTrue(torch.all(diff < 1.0))  # Noise should be small
    
    def test_average_states(self):
        """Test state averaging"""
        # Create test states
        state1 = OrderedDict({
            'param1': torch.tensor([1.0, 2.0, 3.0]),
            'param2': torch.tensor([4.0, 5.0])
        })
        state2 = OrderedDict({
            'param1': torch.tensor([2.0, 4.0, 6.0]),
            'param2': torch.tensor([8.0, 10.0])
        })
        
        # Test uniform averaging
        avg_state = average_states([state1, state2])
        expected_param1 = torch.tensor([1.5, 3.0, 4.5])
        expected_param2 = torch.tensor([6.0, 7.5])
        
        self.assertTrue(torch.allclose(avg_state['param1'], expected_param1))
        self.assertTrue(torch.allclose(avg_state['param2'], expected_param2))
        
        # Test weighted averaging
        weights = [0.7, 0.3]
        weighted_avg = average_states([state1, state2], weights=weights)
        expected_param1 = torch.tensor([1.3, 2.6, 3.9])
        expected_param2 = torch.tensor([5.2, 6.5])
        
        self.assertTrue(torch.allclose(weighted_avg['param1'], expected_param1, atol=1e-6))
        self.assertTrue(torch.allclose(weighted_avg['param2'], expected_param2, atol=1e-6))
    
    def test_plot_curves(self):
        """Test plot generation"""
        curves = {
            'curve1': [1.0, 2.0, 3.0, 4.0],
            'curve2': [4.0, 3.0, 2.0, 1.0]
        }
        
        plot_path = os.path.join(self.temp_dir, 'test_plot.png')
        plot_curves(curves, plot_path, 'Test Plot', 'X', 'Y')
        
        # Check that plot file was created
        self.assertTrue(os.path.exists(plot_path))
    
    def test_split_train_val_from_loader(self):
        """Test train/validation split"""
        # Create mock indices
        indices = list(range(100))
        
        # Create mock loader with indices
        class MockLoader:
            def __init__(self, indices):
                self.indices = indices
        
        loader = MockLoader(indices)
        
        train_idx, val_idx = split_train_val_from_loader(loader, val_fraction=0.2)
        
        # Check split proportions
        self.assertEqual(len(val_idx), 20)  # 20% validation
        self.assertEqual(len(train_idx), 80)  # 80% training
        
        # Check no overlap
        overlap = set(train_idx) & set(val_idx)
        self.assertEqual(len(overlap), 0)
        
        # Check all indices are covered
        all_indices = set(train_idx) | set(val_idx)
        self.assertEqual(all_indices, set(range(100)))

class TestModel(unittest.TestCase):
    
    def test_convnet_mnist(self):
        """Test MNIST ConvNet"""
        model = ConvNetMNIST()
        
        # Test forward pass
        x = torch.randn(2, 1, 28, 28)  # Batch of 2 MNIST images
        output = model(x)
        
        self.assertEqual(output.shape, (2, 10))  # 2 samples, 10 classes
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_convnet_cifar10(self):
        """Test CIFAR-10 ConvNet"""
        model = ConvNetCIFAR10()
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)  # Batch of 2 CIFAR-10 images
        output = model(x)
        
        self.assertEqual(output.shape, (2, 10))  # 2 samples, 10 classes
        self.assertTrue(torch.all(torch.isfinite(output)))
    
    def test_make_model(self):
        """Test model factory function"""
        # Test MNIST
        mnist_model = make_model("MNIST", 10)
        self.assertIsInstance(mnist_model, ConvNetMNIST)
        
        # Test CIFAR-10
        cifar_model = make_model("CIFAR10", 10)
        self.assertIsInstance(cifar_model, ConvNetCIFAR10)
        
        # Test invalid dataset
        with self.assertRaises(ValueError):
            make_model("INVALID", 10)

class TestDataLoader(unittest.TestCase):
    
    def test_set_seed(self):
        """Test seed setting"""
        # Set seed
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Reset seed
        set_seed(42)
        
        # Generate random numbers again
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be identical
        self.assertTrue(torch.allclose(torch_rand, torch_rand2))
        self.assertTrue(np.allclose(np_rand, np_rand2))
    
    def test_dirichlet_non_iid_splits(self):
        """Test non-IID data splitting"""
        # Create mock labels
        labels = [0] * 50 + [1] * 50  # 50 samples of each class
        
        # Test with different alpha values
        splits_low_alpha = dirichlet_non_iid_splits(labels, num_clients=5, alpha=0.1)
        splits_high_alpha = dirichlet_non_iid_splits(labels, num_clients=5, alpha=10.0)
        
        # Check that splits are created
        self.assertEqual(len(splits_low_alpha), 5)
        self.assertEqual(len(splits_high_alpha), 5)
        
        # Check minimum size constraint
        for split in splits_low_alpha:
            self.assertGreaterEqual(len(split), 10)
        
        for split in splits_high_alpha:
            self.assertGreaterEqual(len(split), 10)

if __name__ == '__main__':
    unittest.main()
