# Developer's Guide
This project combines C++ and Python and the strategy is described in this [Blog Post](https://www.hasanfuad.com/post/binding-c-cpp-python/).
It describes the packaging strategy and choices of tools.
To get started with Python packaging, check out the [Python Packaging User Guide](https://packaging.python.org/en/latest/).

## Formatting Style
All codes in this project are formatted with formatting tools. All these formattings are enforced in the CI pipeline.

>[!TIP] If formatting is passing locally but failing in CI, make sure you are using the same version of the formatting tools as specified in the CI pipeline.

- C++ uses `clang-format` with the configuration specified in the [.clang-format](../.clang-format) file.
- CMake files use `cmake-format` with the configuration specified in the [.cmake-format](../.cmake-format.yaml) file.
- Python files uses [`ruff`](https://docs.astral.sh/ruff/formatter/) to format the code with the configuration specified in the [pyproject.toml](../pyproject.toml) file.

### C++ and CMake Formatting
The C++ codes use the [.clang-format](../.clang-format) file and to format the code, you can use `clang-format` tool
with version 18 or later.

Here's a script that can be used to format the C++ code in this project:
```bash
#!/bin/bash

# Function to print colored output
print_colored() {
    local color=$1
    local text=$2

    case $color in
        red)    echo -e "\033[0;31m$text\033[0m" ;;
        green)  echo -e "\033[0;32m$text\033[0m" ;;
        yellow) echo -e "\033[0;33m$text\033[0m" ;;
        *)      echo -e "$text" ;;
    esac
}

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        print_colored "red" "✗ Error: Not in a git repository"
        return 1
    fi
    return 0
}

# Function to check if command exists
command_exists() {
    command -v "$1" > /dev/null 2>&1
}

# Function to check C/C++ files formatting
check_cpp_formatting() {
    local all_good=true

    # Check if clang-format exists
    if ! command_exists clang-format; then
        print_colored "yellow" "⚠ clang-format not found, skipping C/C++ file checks"
        return 0
    fi

    # Get all tracked C/C++ files
    local cpp_files
    cpp_files=$(git ls-files 2>/dev/null | grep -E '\.(c|cpp|cc|cxx|h|hpp|hxx|hh)$' || true)

    if [ -z "$cpp_files" ]; then
        echo "No C/C++ files found in repository"
        return 0
    fi

    print_colored "yellow" "Checking C/C++ files with clang-format..."

    local bad_files=()
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            # Check if file needs formatting
            if ! clang-format --dry-run --Werror "$file" > /dev/null 2>&1; then
                print_colored "red" "✗ $file"
                bad_files+=("$file")
                all_good=false
            fi
        fi
    done <<< "$cpp_files"

    # Store bad files in global variable for fix suggestions
    BAD_CPP_FILES=("${bad_files[@]}")

    
    if [ "$all_good" = true ]; then
        print_colored "green" "✓ All C/C++ files are properly formatted!"
        return 0
    else
        return 1
    fi
}

# Function to check CMake files formatting
check_cmake_formatting() {
    local all_good=true

    # Check if cmake-format exists
    if ! command_exists cmake-format; then
        print_colored "yellow" "⚠ cmake-format not found, skipping CMake file checks"
        return 0
    fi

    # Get all tracked CMake files
    local cmake_files
    cmake_files=$(git ls-files 2>/dev/null | grep -E 'CMakeLists\.txt$|\.cmake$' || true)

    if [ -z "$cmake_files" ]; then
        echo "No CMake files found in repository"
        return 0
    fi

    echo ""
    print_colored "yellow" "Checking CMake files with cmake-format..."

    local bad_files=()
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            # Check if file needs formatting
            if ! cmake-format --check "$file" > /dev/null 2>&1; then
                print_colored "red" "✗ $file"
                bad_files+=("$file")
                all_good=false
            fi
        fi
    done <<< "$cmake_files"

    # Store bad files in global variable for fix suggestions
    BAD_CMAKE_FILES=("${bad_files[@]}")

    if [ "$all_good" = true ]; then
        print_colored "green" "✓ All CMake files are properly formatted!"
        return 0
    else
        return 1
    fi
}

# Function to show fix suggestions
show_fix_suggestions() {
    if [ ${#BAD_CPP_FILES[@]} -gt 0 ] || [ ${#BAD_CMAKE_FILES[@]} -gt 0 ]; then
        echo ""
        print_colored "yellow" "To fix formatting issues:"

        if [ ${#BAD_CPP_FILES[@]} -gt 0 ]; then
            echo ""
            echo "C/C++ files:"
            for file in "${BAD_CPP_FILES[@]}"; do
                echo "  clang-format -i \"$file\""
            done
        fi

        if [ ${#BAD_CMAKE_FILES[@]} -gt 0 ]; then
            echo ""
            echo "CMake files:"
            for file in "${BAD_CMAKE_FILES[@]}"; do
                echo "  cmake-format -i \"$file\""
                            done
        fi

        echo ""
        print_colored "yellow" "Or run all fixes at once:"
        echo "  $0 --fix"
    fi
}

# Function to apply fixes
apply_fixes() {
    local fixed_count=0

    if command_exists clang-format; then
        for file in "${BAD_CPP_FILES[@]}"; do
            if [ -f "$file" ]; then
                clang-format -i "$file"
                print_colored "green" "✓ Fixed: $file"
                ((fixed_count++))
            fi
        done
    fi

    if command_exists cmake-format; then
        for file in "${BAD_CMAKE_FILES[@]}"; do
            if [ -f "$file" ]; then
                cmake-format -i "$file"
                print_colored "green" "✓ Fixed: $file"
                ((fixed_count++))
            fi
        done
    fi

    if [ $fixed_count -gt 0 ]; then
        print_colored "green" "\n✓ Fixed $fixed_count file(s)"
    fi
}

# Main function
main() {
    # Global arrays to store bad files
    declare -a BAD_CPP_FILES
    declare -a BAD_CMAKE_FILES

    # Check for help or fix argument
    if [[ "$1" == "--help" || "$1" == "-h" ]]; then
        echo "Usage: $0 [--fix | -h | --help]"
        echo ""
        echo "Check code formatting in a git repository."
        echo ""
        echo "Options:"
        echo "  --fix     Apply fixes to improperly formatted files"
        echo "  -h, --help Show this help message"
        return 0
    fi

    # Check if we're in a git repository
    if ! check_git_repo; then
        return 1
    fi

    # Run checks
    local cpp_ok=true
    local cmake_ok=true

    if ! check_cpp_formatting; then
        cpp_ok=false
    fi

    if ! check_cmake_formatting; then
        cmake_ok=false
            fi

    if ! check_cmake_formatting; then
        cmake_ok=false
    fi

    # Show summary
    echo ""
    if [ "$cpp_ok" = true ] && [ "$cmake_ok" = true ]; then
        print_colored "green" "All files are properly formatted!"
        return 0
    else
        show_fix_suggestions

        # Apply fixes if requested
        if [[ "$1" == "--fix" ]]; then
            echo ""
            print_colored "yellow" "Applying fixes..."
            apply_fixes
        fi

        return 1
    fi
}

# Run main function with all arguments
main "$@"
```

It will check both C++ and CMake files and print out the files that are not properly formatted.

Run the following to check the formatting:
```bash
./check-format.sh
```
and if you want to apply the fixes, run:
```bash
./check-format.sh --fix
```

### Python Formatting
The Python codes use `ruff` to format as well as lint the code.

To check the formatting, run:
```bash
ruff format --check
```
and it will print the number of files that are not properly formatted. To apply the formatting, run:
```bash
ruff format
```

And lint the code with:
```bash
ruff check
```

## Installation
Check the [README.md](../README.md) file for installation instructions. For development, install from source.
To run the tests, install `pytest` and do:
```bash
python -m pip install -e .[test]
```
and run
```bash
pytest
```

## Packaging
### Create Distribution Files
After development and testing, to create the archive and wheel for distribution, install `build` using `pip` and run:

```bash
python -m build
```
which will create the distribution files in the `dist/` directory.

### Upload to PyPI/TestPyPI
To upload the distribution files to PyPI or TestPyPI, install `twine` using `pip` and run:
```bash
python -m twine upload --repository testpypi dist/*
```

It will require a `testpypi` `index-server` entry in your `~/.pypirc` file. See PyPI documentation
for more details.
