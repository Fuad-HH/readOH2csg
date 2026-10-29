# The Python suite compares outputs with the saved results in tests/assets.
# It needs the package installed (pip install -e .[test]).
find_package(Python REQUIRED COMPONENTS Interpreter)
add_test(
  NAME pytest
  COMMAND ${Python_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/python -q
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
