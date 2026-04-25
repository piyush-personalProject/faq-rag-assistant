# Contributing to Knowledge Store

Thank you for your interest in contributing to Knowledge Store!

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/knowledgeStore.git
   cd knowledgeStore
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Install pre-commit hooks**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix-name
```

### 2. Make Your Changes

- Write code following our [Code Style](#code-style)
- Add tests for new functionality
- Update documentation as needed
- Follow our [commit message format](#commit-messages)

### 3. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/test_rag_engine.py -v
```

### 4. Run Pre-commit

```bash
pre-commit run --all-files
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "feat: add new feature"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub.

## Code Style

- Follow **PEP 8** style guidelines
- Use **type hints** for all function signatures
- Maximum line length: **100 characters**
- Use **4 spaces** for indentation

### Python Tools

We use these tools (configured in `.pre-commit-config.yaml`):

| Tool | Purpose |
|------|---------|
| `black` | Code formatting |
| `isort` | Import sorting |
| `flake8` | Linting |
| `mypy` | Type checking |

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Formatting, missing semicolons, etc |
| `refactor` | Code restructuring without behavior change |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |

### Examples

```
feat(rag_engine): add hybrid search with BM25
fix(graph_rag): resolve quality check timeout
docs(readme): update installation instructions
```

## Testing Guidelines

- All new features require tests
- All bug fixes require a test that reproduces the bug
- Maintain test coverage above 80%
- Tests should be independent and reproducible

### Test Structure

```python
def test_feature_name():
    """Description of what is being tested."""
    # Arrange
    ...
    # Act
    ...
    # Assert
    ...
```

## Pull Request Process

1. Fill out the PR template completely
2. Link any related issues
3. Ensure all CI checks pass
4. Request review from maintainers
5. Be responsive to feedback

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
