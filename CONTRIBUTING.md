# Contributing to MeVe Framework

Thank you for your interest in contributing to MeVe! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nakulbh/Meve-framework.git
   cd meve
   ```

2. **Install dependencies**
   ```bash
   make setup
   # or with dev dependencies
   make install-dev
   ```

3. **Download sample data**
   ```bash
   make download-data
   ```

## Project Structure

```
meve/
├── meve/              # Core package
│   ├── core/         # Engine and models
│   ├── phases/       # 5 pipeline phases
│   ├── services/     # External services
│   └── utils/        # Utilities
├── integrations/     # MCP, agents, API
├── tests/           # Test suite
├── examples/        # Usage examples
├── scripts/         # Dev scripts
└── config/          # Configuration files
```

## Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new functionality

3. **Run tests and linting**
   ```bash
   make test
   make lint
   make format
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: description of your changes"
   ```

5. **Push and create a PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## Commit Message Guidelines

Follow conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write docstrings for all public functions/classes
- Keep functions focused and small
- Use meaningful variable names

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Test edge cases

### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration
```

## Adding New Features

### Adding a New Phase

1. Create file in `meve/phases/`
2. Implement phase logic
3. Add to `meve/phases/__init__.py`
4. Update engine to use new phase
5. Add tests

### Adding a New Integration

1. Create directory in `integrations/`
2. Implement integration logic
3. Add README.md with usage
4. Add example
5. Add integration tests

## Questions?

Open an issue or reach out to the maintainers!

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
