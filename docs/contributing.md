# Contributing to MM-GRPO

We welcome contributions of any kind: bug fixes, enhancements, documentation improvements, or feedback.

## Ways to Contribute

Your support can take many forms:

- **Report Issues**: Report bugs or unexpected behaviors in [Issues](https://github.com/leibniz-csi/mm_grpo/issues)
- **Suggest Features**: Propose new features in [Issues](https://github.com/leibniz-csi/mm_grpo/issues)
- **Submit Code**: Implement features or fixes via pull requests
- **Review PRs**: Help review pull requests and assist other contributors
- **Share**: Share MM-GRPO in blog posts, social media, or give the repo a ⭐

## Finding Issues to Contribute

Looking for ways to dive in? Check out:

- [Roadmap Issues](https://github.com/leibniz-csi/mm_grpo/issues?q=is%3Aissue+state%3Aopen+label%3Aroadmap)

- [Good First Issues](https://github.com/leibniz-csi/mm_grpo/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)

- [Bug Reports](https://github.com/leibniz-csi/mm_grpo/issues?q=is%3Aissue+is%3Aopen+label%3Abug)

## Getting Started

### 1. Fork and Clone

Fork the `mm_grpo` repository on GitHub, then clone your fork:

```bash
git clone git@github.com:your_name_here/mm_grpo.git
cd mm_grpo
```

Add the official repository as upstream:

```bash
git remote add upstream git@github.com:leibniz-csi/mm_grpo.git
```

### 2. Install Development Environment

Install your local copy into your environment. See the [Installation Guide](installation.md) for details.

### 3. Create a Branch

Create a branch for your changes:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

### 4. Make Changes

Make your changes locally. Follow these guidelines:

- **Code Style**: Follow PEP 8 and use consistent formatting
- **Documentation**: Update docs for user-facing changes
- **Tests**: Add tests for new functionality
- **Type Hints**: Add type hints to function signatures
- **Docstrings**: Document classes and functions using Google-style docstrings

### 5. Run Linters and Tests

Before submitting, ensure your code passes all checks:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Run pre-commit hooks
pre-commit run --show-diff-on-failure --color=always --all-files
```

### 6. Run Tests

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-asyncio
```

Run the test suite:

```bash
python3 -m pytest --cov=gerl/workers tests/workers
```

### 7. Commit and Push

Commit your changes with a clear message:

```bash
git add .
git commit -m "Your detailed description of your changes"
git push origin name-of-your-bugfix-or-feature
```

### 8. Submit a Pull Request

Submit a pull request through the GitHub website. Include:

- Clear description of changes
- Reference to related issues
- Screenshots (if UI changes)
- Test results

## Pull Request Guidelines

To streamline reviews:

- ✅ Adhere to pre-commit lint rules
- ✅ Ensure all checks pass
- ✅ Update documentation for user-facing changes
- ✅ Add or update tests for new functionality
- ✅ Keep PRs focused and reasonably sized
- ✅ Write clear commit messages

## Code Style

- Follow PEP 8 style guide
- Use type hints where possible
- Write clear, descriptive variable and function names
- Add comments for complex logic
- Keep functions focused and reasonably sized

## Documentation

When contributing:

- Update relevant documentation pages
- Add docstrings to new functions/classes
- Include examples for new features
- Update the changelog for significant changes

## Testing

- Add unit tests for new functionality
- Ensure existing tests still pass
- Test edge cases and error conditions
- Aim for good test coverage

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

Thank you for contributing to MM-GRPO! 🎉
