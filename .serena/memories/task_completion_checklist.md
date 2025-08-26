# Task Completion Checklist

## When a Development Task is Completed

### Code Quality Checks
- [ ] **Type Hints**: Ensure all functions have proper type annotations
- [ ] **Error Handling**: Proper exception handling with meaningful error messages
- [ ] **Documentation**: Functions and classes have appropriate docstrings
- [ ] **Code Style**: Follows established conventions (snake_case, pathlib, etc.)

### Testing Requirements
- [ ] **Functionality**: Core functionality works as expected
- [ ] **Edge Cases**: Handle edge cases appropriately
- [ ] **Error Conditions**: Proper error handling and recovery

### PackRepo Specific Requirements
When implementing PackRepo components:

- [ ] **Determinism**: --no-llm mode produces identical outputs across runs
- [ ] **Budget Compliance**: Token budgets never exceeded, hard caps enforced
- [ ] **Security**: Respects .gitignore, license headers, secret scanning
- [ ] **Performance**: Meets latency requirements (p50 ≤ +30%, p95 ≤ +50% of baseline)
- [ ] **Memory**: Stays within 8GB RAM limit
- [ ] **Output Format**: Single UTF-8 file with JSON index + body sections
- [ ] **Line Anchors**: Preserves line anchors for every included span

### Environment Validation
- [ ] **Python Version**: Runs on Python 3.10+
- [ ] **Dependencies**: All required packages installed and working
- [ ] **Environment Reproducibility**: Requirements pinned, seeds set for determinism

### Documentation Updates
- [ ] **README**: Updated if public API changes
- [ ] **TODO**: Mark completed items, update progress
- [ ] **Comments**: In-code documentation for complex logic

### Version Control
- [ ] **Commit Message**: Clear, descriptive commit messages
- [ ] **Git Status**: Clean working directory
- [ ] **Branch**: Work on appropriate feature branch if applicable

### Integration Points
- [ ] **Library Interface**: Minimal surface between rendergit.py and PackRepo library
- [ ] **Backward Compatibility**: Existing rendergit functionality preserved
- [ ] **API Stability**: Public interfaces remain stable