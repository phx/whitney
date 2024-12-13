"""Generate pre-trained trigger classifier weights."""

import numpy as np
import os

def generate_trigger_weights():
    """Generate and save trigger classifier weights."""
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate example weights (would normally be trained)
    W = np.random.randn(64, 32) * 0.1
    
    # Save weights
    np.save('models/trigger_classifier.npy', W)

if __name__ == '__main__':
    generate_trigger_weights() 