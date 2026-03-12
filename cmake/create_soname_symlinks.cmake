# cmake/create_soname_symlinks.cmake
# Creates SONAME copies for all shared libraries in LIB_DIR so that the
# dynamic linker can resolve SONAME references at runtime.
# Usage: cmake -DLIB_DIR=<path> -DPATCHELF=<path> -P create_soname_symlinks.cmake

file(GLOB _libs "${LIB_DIR}/*.so" "${LIB_DIR}/*.so.*")
foreach(_lib ${_libs})
  get_filename_component(_fname ${_lib} NAME)
  execute_process(
    COMMAND ${PATCHELF} --print-soname ${_lib}
    OUTPUT_VARIABLE _soname
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _result)
  if(_result EQUAL 0
     AND _soname
     AND NOT _soname STREQUAL _fname)
    set(_soname_path "${LIB_DIR}/${_soname}")
    if(NOT EXISTS ${_soname_path})
      message(STATUS "Creating SONAME copy: ${_soname} (from ${_fname})")
      file(COPY_FILE ${_lib} ${_soname_path})
    endif()
  endif()
endforeach()
