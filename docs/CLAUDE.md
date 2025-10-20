# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Development Commands

**Package Management:**
```bash
uv sync -U                     # Install dependencies
uv add <package>               # Add new dependency
```

**Linting and Formatting:**
```bash
uv run ruff check src/          # Run linting checks
uv run ruff format src/         # Format code
uv run ruff check --fix src/    # Auto-fix linting issues
```

**Testing:**
```bash
# Basic execution
uv run pytest                   # Run all unit tests (excludes hardware/GUI/WebUI by default)
uv run pytest -v                # Verbose output
uv run pytest --cov=oiwalab_meas --cov-report=html  # Generate coverage report

# Test selection
uv run pytest -m "unit"         # Run only unit tests
uv run pytest -m "integration"  # Run only integration tests
uv run pytest -k "test_config"  # Run tests matching pattern

# Development workflows
uv run pytest --lf              # Run last failed tests only
uv run pytest tests/unit/core/test_drivers.py::TestQDAC2CustomDriver  # Specific test
```


**YAML Formatting:**
Uses ruamel.yaml for round-trip preservation:
- Numeric lists: Flow style `[1, 3, 5]`
- Complex structures: Block style with indentation
- Anchors & Aliases: Fully preserved across editing (`&ref`, `*ref`)
- WebUI compatibility: Structured interface preserves all formatting

### Module Organization

## Development Principles

### Early Development Stage
**Backward compatibility is not a priority.** Breaking changes are acceptable for better design.

Core principles:
- Prioritize optimal design over compatibility
- Refactor aggressively when architecture improves
- Fix fundamental issues early
- Focus on long-term maintainability

### YAGNI (You Aren't Gonna Need It)
- Start with minimal functionality
- Implement solutions only when problems arise
- Avoid over-abstraction
- Consider abstraction only after seeing the same pattern 3 times

### Implementation Guidelines
1. Write minimal working code first
2. Improve only when problems occur
3. Keep documentation minimal and focused
4. Don't hesitate to make breaking changes for better design

## Code Style

Follow rules defined in `pyproject.toml`:

**Formatting (Ruff):**
- Python 3.12+ required
- Line length: 88 characters
- Double quotes
- 4 spaces indentation

**Linting:**
Extensive rule set including pycodestyle, Pyflakes, pyupgrade, flake8-bugbear, isort, Pylint, and more.

**Special Allowances:**
- Type hints encouraged but not strictly enforced


## Documentation Guidelines

When writing documentation:

1. **Be concise and factual** - Avoid promotional language
2. **Use objective descriptions** - State what something does
3. **Avoid marketing terms** - "comprehensive", "advanced", "powerful", "cutting-edge"
4. **Remove unnecessary emphasis** - Minimize bold, ALL CAPS, exclamation marks
5. **Focus on technical accuracy** - Describe functionality
6. **Keep tone neutral** - Technical documentation, not advertising

**Examples:**

❌ Bad: "**PRIMARY INTERFACE** - Comprehensive unified configuration management with advanced validation"
✅ Good: "Unified configuration management with validation"

❌ Bad: "Cutting-edge real-time plotting interface with powerful FFT capabilities"
✅ Good: "Real-time plotting interface with FFT support"