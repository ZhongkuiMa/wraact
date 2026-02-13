# Contributing to WRAACT

Thank you for your interest in contributing to WRAACT! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YourUsername/wraact.git
   cd wraact
   ```

2. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Verify Installation**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the code style guidelines below
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Quality Checks**
   ```bash
   # Run tests
   pytest tests/ -v

   # Lint code
   ruff check src/wraact tests

   # Type checking
   mypy src/wraact
   ```

## Code Quality Standards

### Linting
```bash
ruff check src/wraact tests
```

### Formatting
```bash
ruff format src/wraact tests
```

### Type Checking
```bash
mypy src/wraact
```

### Testing
```bash
pytest tests/ -v
```

## Pull Request Guidelines

Before submitting a pull request:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Code passes linting (`ruff check src/wraact tests`)
- [ ] Type checking passes (`mypy src/wraact`)
- [ ] New functionality includes tests
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

## Testing

- **Run all tests**: `pytest tests/ -v`
- **Run with coverage**: `pytest tests/ --cov=wraact --cov-report=term-missing`
- **Current stats**: 1876 tests, 92% coverage

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Keep functions focused and well-documented
- Use clear, descriptive variable names

## CI/CD

All pull requests trigger GitHub Actions that run:
- Unit tests across multiple Python versions
- Code quality checks (ruff, mypy)
- Coverage reporting

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bug Reports**: Open a GitHub Issue with the bug template
- **Feature Requests**: Open a GitHub Issue with the feature request template
