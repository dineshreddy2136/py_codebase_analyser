# Codebase Dependency Analyzer

A comprehensive Python application that analyzes function dependencies in codebases provided as `.whl` or `.zip` files. Given a function snippet, it extracts and identifies all dependency functions and outputs them in the correct execution order.

## Features

- **Archive Support**: Process `.whl` and `.zip` codebase files
- **Fast File Search**: Efficient file and content search with pattern matching
- **Dependency Analysis**: Deep analysis of function dependencies using Python AST
- **Correct Ordering**: Topological sorting to determine proper dependency execution order
- **Multi-language Ready**: Extensible architecture for supporting multiple programming languages
- **Caching**: Built-in caching for improved performance on large codebases
- **Detailed Output**: Comprehensive information about each dependency including source code and documentation

## Installation

The application is contained in a single Python file with minimal dependencies:

### Required Dependencies
```bash
# Core Python (no additional packages required for basic functionality)
python >= 3.7
```

### Optional Dependencies (for enhanced features)
```bash
# For .whl file support
pip install wheel

# For advanced parsing (future enhancement)
pip install tree-sitter tree-sitter-python tree-sitter-javascript
```

## Usage

### Basic Usage
```bash
python codebase_dependency_analyzer.py <codebase.zip> <function_snippet.txt>
```

### Examples
```bash
# Analyze a Python package
python codebase_dependency_analyzer.py mypackage.whl function_snippet.txt

# Analyze a zipped codebase
python codebase_dependency_analyzer.py codebase.zip snippet.py

# Output to file
python codebase_dependency_analyzer.py codebase.zip snippet.txt -o results.txt

# JSON output format
python codebase_dependency_analyzer.py codebase.zip snippet.txt --format json
```

### Command Line Options
- `codebase`: Path to `.whl` or `.zip` file containing the codebase
- `snippet`: Path to text file containing the function snippet to analyze
- `--output`, `-o`: Output file for results (default: stdout)
- `--format`: Output format - `text` (default) or `json`

## Function Snippet Format

The function snippet file should contain the function you want to analyze. It can be:

1. **Complete function definition**:
```python
def my_function(a, b):
    result = helper_function(a)
    return process_result(result, b)
```

2. **Partial function snippet**:
```python
def complex_operation(self, a, b, c):
    temp1 = self.add(a, b)
    temp2 = self.multiply(temp1, c)
    return main_function(temp2, temp1)
```

3. **Just the function signature**:
```python
def target_function(param1, param2):
```

## Example Output

```
Codebase Dependency Analysis Results
========================================

Function found: complex_operation
File: example_code.py
Line: 24
Search method: function_definition

Dependencies (6 total):
------------------------------
1. helper_function
2. Calculator.add
3. another_helper
4. Calculator.multiply
5. main_function
6. Calculator.complex_operation

Detailed Dependency Information:
-----------------------------------

Function: helper_function
  File: example_code.py
  Line: 3
  Signature: helper_function(x)
  Docstring: A simple helper function
  Source code (3 lines):
    1: def helper_function(x):
    2:     """A simple helper function"""
    3:     return x * 2

[... more dependencies ...]
```

## Architecture

The application consists of several key components:

### Core Engines
- **`FileSearchEngine`**: Fast file discovery and pattern matching
- **`ContentSearchEngine`**: Content search with regex support
- **`PythonDependencyAnalyzer`**: AST-based dependency analysis for Python code
- **`CodebaseExtractor`**: Archive extraction and management

### Key Classes
- **`FunctionInfo`**: Stores complete function metadata
- **`FileMatch`**: Represents search results
- **`SearchResult`**: Aggregated search operation results

### Analysis Process
1. **Extract**: Unzip/extract the codebase archive
2. **Search**: Locate the target function using the provided snippet
3. **Analyze**: Parse Python files using AST to build dependency graph
4. **Order**: Apply topological sorting for correct execution order
5. **Report**: Generate detailed dependency information

## How It Works

1. **Codebase Extraction**: Archives are extracted to temporary directories
2. **Function Discovery**: The target function is located using pattern matching
3. **AST Analysis**: Python files are parsed to build a complete function dependency graph
4. **Dependency Resolution**: Recursive dependency discovery with cycle detection
5. **Topological Sorting**: Dependencies are ordered using Kahn's algorithm
6. **Detailed Reporting**: Complete function information including source code and documentation

## Supported Languages

Currently supports:
- **Python** (full support with AST analysis)

Extensible architecture allows for easy addition of:
- JavaScript/TypeScript
- Java
- C/C++
- Other languages via tree-sitter integration

## Performance Features

- **Caching**: File system crawl results are cached for repeated analysis
- **Parallel Processing**: Concurrent file processing for large codebases
- **Smart Filtering**: Efficient ignore patterns to skip irrelevant files
- **Memory Optimization**: Streaming file processing to handle large archives

## Error Handling

The application provides comprehensive error handling for:
- Invalid or corrupted archives
- Malformed function snippets
- Syntax errors in source files
- Missing dependencies
- Circular dependency detection

## Limitations

- Currently focused on Python codebases (extensible to other languages)
- Dynamic dependencies (runtime imports) are not fully supported
- Some complex metaprogramming patterns may not be detected

## Testing

The repository includes example files for testing:
- `example_code.py`: Sample Python code with dependencies
- `function_snippet.txt`: Sample function snippet
- `example_codebase.zip`: Test archive

Run the test:
```bash
python codebase_dependency_analyzer.py example_codebase.zip function_snippet.txt
```

## Contributing

The application is designed to be extensible. Key areas for contribution:
- Additional language support
- Enhanced dependency detection algorithms
- Performance optimizations
- UI improvements

## License

This project incorporates concepts from various open-source search and analysis tools while providing a unified dependency analysis solution.
