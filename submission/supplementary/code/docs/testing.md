# Testing Guide

This document describes the testing framework and procedures for the Fractal Field Theory codebase.

## 1. Test Organization

### 1.1 Directory Structure
```
tests/
├── conftest.py          # Common fixtures and configuration
├── pytest.ini          # Test configuration
├── test_field.py       # UnifiedField tests
├── test_basis.py       # FractalBasis tests
├── test_detector.py    # Detector tests
└── test_types.py       # Type system tests
```

### 1.2 Test Categories
- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Numerical Tests**: Test precision and stability
- **Performance Tests**: Test computation efficiency

## 2. Running Tests

### 2.1 Basic Usage
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_field.py

# Run tests by marker
python -m pytest -m "not slow"
```

### 2.2 Coverage Reports
```bash
# Generate coverage report
python -m pytest --cov=core

# Generate HTML coverage report
python -m pytest --cov=core --cov-report=html
```

## 3. Writing Tests

### 3.1 Using Fixtures
```python
def test_field_evolution(field, energy_points):
    """Example using common fixtures."""
    result = field.evolve(energy_points)
    assert np.all(np.isfinite(result))
```

### 3.2 Test Categories
```python
@pytest.mark.slow
def test_intensive_computation():
    """Long-running test."""
    pass

@pytest.mark.integration
def test_component_interaction():
    """Integration test."""
    pass

@pytest.mark.numerical
def test_precision():
    """Numerical precision test."""
    pass
```

## 4. Test Configuration

### 4.1 Coverage Requirements
- Minimum coverage: 80%
- Excluded patterns:
  - `__repr__` methods
  - `NotImplementedError` raises
  - `if __name__ == '__main__'` blocks

### 4.2 Custom Markers
- `slow`: Long-running tests
- `integration`: Component interaction tests
- `numerical`: Precision-sensitive tests

## 5. Best Practices

### 5.1 Test Structure
1. Arrange: Set up test conditions
2. Act: Execute the test
3. Assert: Verify results

### 5.2 Naming Conventions
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### 5.3 Documentation
- Clear test descriptions
- Document test assumptions
- Explain complex assertions

## 6. Common Patterns

### 6.1 Testing Numerical Results
```python
def test_numerical_computation():
    """Test numerical results."""
    result = compute_value()
    assert np.isclose(result, expected, rtol=1e-6)
```

### 6.2 Testing Exceptions
```python
def test_invalid_input():
    """Test error handling."""
    with pytest.raises(ValueError):
        process_input(-1)
```

### 6.3 Testing Async Code
```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test asynchronous operations."""
    result = await async_compute()
    assert result is not None
```

## 7. Troubleshooting

### 7.1 Common Issues
1. Floating point comparisons
2. Random number generation
3. File path handling
4. Memory management

### 7.2 Solutions
1. Use `np.isclose()` for float comparisons
2. Set random seeds in fixtures
3. Use `pathlib` for paths
4. Clean up resources in fixtures 