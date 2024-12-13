# Testing Guide

## Test Organization

Tests are organized by functionality and marked with appropriate markers:

- `@pytest.mark.physics`: Physics-related tests
- `@pytest.mark.numeric`: Numerical computation tests
- `@pytest.mark.performance`: Performance and benchmarking tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.unit`: Unit tests

## Coverage Requirements

All code must maintain at least 80% test coverage. Coverage reports are generated in HTML format.

+ ### Coverage Configuration
+ - Minimum coverage threshold: 80%
+ - HTML reports generated in htmlcov/
+ - Excluded patterns:
+   - `__repr__` methods
+   - Debug logging
+   - Platform-specific code
+ 
+ ### Coverage Reports
+ Coverage reports show:
+ - Line coverage
+ - Branch coverage
+ - Function coverage
+ - Module coverage

## Test Fixtures

Common test fixtures are defined in `tests/conftest.py`:

- `test_grid`: Standard spatial grid for testing
- `energy_points`: Standard energy points from Z mass to Planck mass
- `detector`: Configured detector instance
- `test_data_generator`: Function to generate test data
- `test_covariance`: Function to generate covariance matrices
- `numeric_precision`: Numerical precision requirements
- `test_config`: Basic test configuration
- `field_config`: Standard field configuration

+ ### Using Fixtures
+ ```python
+ def test_field_evolution(test_grid, energy_points):
+     """Example using test fixtures."""
+     # Test grid provides spatial points
+     field = compute_field(test_grid)
+     
+     # Energy points for evolution
+     evolved = field.evolve(energy_points)
+     
+     assert is_valid_evolution(evolved)
+ ```

## Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest -m physics
python -m pytest -m "not slow"
python -m pytest -m "integration or unit"

# Generate coverage report
python -m pytest --cov=core --cov-report=html
```

+ ## Best Practices
+ 
+ ### Test Structure
+ 1. Arrange: Set up test conditions
+ 2. Act: Execute the test
+ 3. Assert: Verify results
+ 
+ ### Naming Conventions
+ - Test files: `test_*.py`
+ - Test classes: `Test*`
+ - Test functions: `test_*`
+ 
+ ### Documentation
+ - Clear test descriptions
+ - Document test assumptions
+ - Explain complex assertions