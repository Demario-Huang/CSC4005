# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.21

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/cmake/bin/cmake

# The command to remove a file.
RM = /opt/cmake/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/host/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/host/code/build

# Include any dependencies generated for this target.
include CMakeFiles/main_3.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main_3.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main_3.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main_3.dir/flags.make

CMakeFiles/main_3.dir/src/main_3.cpp.o: CMakeFiles/main_3.dir/flags.make
CMakeFiles/main_3.dir/src/main_3.cpp.o: ../src/main_3.cpp
CMakeFiles/main_3.dir/src/main_3.cpp.o: CMakeFiles/main_3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/host/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main_3.dir/src/main_3.cpp.o"
	/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main_3.dir/src/main_3.cpp.o -MF CMakeFiles/main_3.dir/src/main_3.cpp.o.d -o CMakeFiles/main_3.dir/src/main_3.cpp.o -c /mnt/host/code/src/main_3.cpp

CMakeFiles/main_3.dir/src/main_3.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main_3.dir/src/main_3.cpp.i"
	/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/host/code/src/main_3.cpp > CMakeFiles/main_3.dir/src/main_3.cpp.i

CMakeFiles/main_3.dir/src/main_3.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main_3.dir/src/main_3.cpp.s"
	/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/host/code/src/main_3.cpp -o CMakeFiles/main_3.dir/src/main_3.cpp.s

CMakeFiles/main_3.dir/src/graphic.cpp.o: CMakeFiles/main_3.dir/flags.make
CMakeFiles/main_3.dir/src/graphic.cpp.o: ../src/graphic.cpp
CMakeFiles/main_3.dir/src/graphic.cpp.o: CMakeFiles/main_3.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/host/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main_3.dir/src/graphic.cpp.o"
	/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main_3.dir/src/graphic.cpp.o -MF CMakeFiles/main_3.dir/src/graphic.cpp.o.d -o CMakeFiles/main_3.dir/src/graphic.cpp.o -c /mnt/host/code/src/graphic.cpp

CMakeFiles/main_3.dir/src/graphic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main_3.dir/src/graphic.cpp.i"
	/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/host/code/src/graphic.cpp > CMakeFiles/main_3.dir/src/graphic.cpp.i

CMakeFiles/main_3.dir/src/graphic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main_3.dir/src/graphic.cpp.s"
	/opt/llvm/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/host/code/src/graphic.cpp -o CMakeFiles/main_3.dir/src/graphic.cpp.s

# Object files for target main_3
main_3_OBJECTS = \
"CMakeFiles/main_3.dir/src/main_3.cpp.o" \
"CMakeFiles/main_3.dir/src/graphic.cpp.o"

# External object files for target main_3
main_3_EXTERNAL_OBJECTS =

main_3: CMakeFiles/main_3.dir/src/main_3.cpp.o
main_3: CMakeFiles/main_3.dir/src/graphic.cpp.o
main_3: CMakeFiles/main_3.dir/build.make
main_3: libimgui.a
main_3: /usr/lib64/libfreetype.so
main_3: /usr/lib64/libSDL2.so
main_3: /usr/lib64/libGLX.so
main_3: /usr/lib64/libOpenGL.so
main_3: CMakeFiles/main_3.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/host/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable main_3"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main_3.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main_3.dir/build: main_3
.PHONY : CMakeFiles/main_3.dir/build

CMakeFiles/main_3.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main_3.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main_3.dir/clean

CMakeFiles/main_3.dir/depend:
	cd /mnt/host/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/host/code /mnt/host/code /mnt/host/code/build /mnt/host/code/build /mnt/host/code/build/CMakeFiles/main_3.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/main_3.dir/depend
