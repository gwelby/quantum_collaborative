# Publishing Quantum Field Visualization to PyPI

This guide will walk you through the process of publishing the Quantum Field Visualization package to the Python Package Index (PyPI), making it available for installation via pip.

## Prerequisites

Before publishing to PyPI, ensure you have the following:

1. A PyPI account (create one at [pypi.org](https://pypi.org/account/register/))
2. The latest version of these tools:
   - `pip`
   - `build`
   - `twine`

```bash
python -m pip install --upgrade pip build twine
```

## Preparation Steps

### 1. Update Package Version

Update the version number in `quantum_field/version.py`:

```python
__version__ = "0.2.0"  # Update this to the new version number
```

### 2. Update README and Documentation

Ensure the README.md and documentation are up-to-date:

- Verify installation instructions
- Update API references if needed
- Check that example code works with the latest version

### 3. Update CHANGELOG

If not already present, create or update a CHANGELOG.md file:

```markdown
# Changelog

## 0.2.0 (YYYY-MM-DD)

### Added
- Feature 1
- Feature 2

### Changed
- Change 1
- Change 2

### Fixed
- Fix 1
- Fix 2

## 0.1.0 (Previous release date)

Initial release
```

### 4. Verify Package Structure

Ensure your package structure is correct:

```
quantum_field/
├── __init__.py
├── version.py
├── core.py
└── ... (other modules)
setup.py
README.md
LICENSE
```

### 5. Check setup.py Configuration

Ensure setup.py is correctly configured:

- Package name
- Version (imported from `quantum_field/version.py`)
- Description
- Long description (from README.md)
- Author information
- Dependencies
- Classifiers
- Python version requirements

## Building the Distribution Packages

### 1. Clean Previous Builds

Remove any previous build artifacts:

```bash
rm -rf build/ dist/ *.egg-info/
```

### 2. Build the Package

Build the source distribution and wheel:

```bash
python -m build
```

This will create:
- A source distribution (`dist/quantum_field-X.Y.Z.tar.gz`)
- A wheel distribution (`dist/quantum_field-X.Y.Z-py3-none-any.whl`)

### 3. Check the Built Packages

Verify the packages are built correctly:

```bash
# List files in the wheel
python -m pip install dist/*.whl --dry-run

# Check the built distributions
twine check dist/*
```

## Testing before Publishing

### 1. Test in a Virtual Environment

Test the built package in a clean virtual environment:

```bash
# Create a virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\\Scripts\\activate

# Install the wheel
pip install dist/*.whl

# Run tests to ensure the installed package works
python -c "from quantum_field import generate_quantum_field; print(generate_quantum_field(10, 10, 'love').shape)"

# Exit the virtual environment
deactivate
```

### 2. Test with TestPyPI (Optional)

Test publishing to TestPyPI before the official release:

```bash
# Upload to TestPyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Test installation from TestPyPI
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ quantum-field
```

## Publishing to PyPI

### 1. Upload to PyPI

Once testing is complete, upload to the official PyPI:

```bash
twine upload dist/*
```

You'll be prompted for your PyPI username and password.

### 2. Verify the Published Package

Check that the package is available on PyPI:

1. Visit https://pypi.org/project/quantum-field/
2. Test installation in a clean environment:

```bash
pip install quantum-field
```

## Automating Future Releases

Consider setting up automated releases using GitHub Actions:

1. Create a workflow file at `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [created]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.x'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    - name: Build and publish
      env:
        TWINE_USERNAME: ${{ secrets.PYPI_USERNAME }}
        TWINE_PASSWORD: ${{ secrets.PYPI_PASSWORD }}
      run: |
        python -m build
        twine upload dist/*
```

2. Store your PyPI credentials as GitHub secrets.
3. Future releases can be published by creating a release in GitHub.

## Post-Publication Tasks

After publishing:

1. Update documentation to reflect the new version
2. Announce the release on relevant channels
3. Tag the release in the git repository:

```bash
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

4. Start planning the next version

## Troubleshooting

### Common Issues

1. **Version conflict**: Ensure you don't publish a version that already exists on PyPI.
2. **Missing dependencies**: Verify all dependencies are correctly listed in setup.py.
3. **README not rendering**: Ensure your README.md uses valid Markdown syntax.
4. **Permission denied**: Check that you have the correct PyPI credentials.
5. **Package not found after publishing**: It may take a few minutes for the package to become available.

### Getting Help

If you encounter issues with PyPI publishing:

- Check the [PyPI documentation](https://packaging.python.org/tutorials/packaging-projects/)
- Review the [Python Packaging User Guide](https://packaging.python.org/)
- Check the Twine documentation for upload-specific issues