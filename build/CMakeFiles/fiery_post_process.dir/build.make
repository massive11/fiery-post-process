# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kismet/cppProjects/fiery_post_process

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kismet/cppProjects/fiery_post_process/build

# Include any dependencies generated for this target.
include CMakeFiles/fiery_post_process.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fiery_post_process.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fiery_post_process.dir/flags.make

CMakeFiles/fiery_post_process.dir/Hungarian.cpp.o: CMakeFiles/fiery_post_process.dir/flags.make
CMakeFiles/fiery_post_process.dir/Hungarian.cpp.o: ../Hungarian.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kismet/cppProjects/fiery_post_process/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fiery_post_process.dir/Hungarian.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fiery_post_process.dir/Hungarian.cpp.o -c /home/kismet/cppProjects/fiery_post_process/Hungarian.cpp

CMakeFiles/fiery_post_process.dir/Hungarian.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fiery_post_process.dir/Hungarian.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kismet/cppProjects/fiery_post_process/Hungarian.cpp > CMakeFiles/fiery_post_process.dir/Hungarian.cpp.i

CMakeFiles/fiery_post_process.dir/Hungarian.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fiery_post_process.dir/Hungarian.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kismet/cppProjects/fiery_post_process/Hungarian.cpp -o CMakeFiles/fiery_post_process.dir/Hungarian.cpp.s

CMakeFiles/fiery_post_process.dir/fiery_process.cpp.o: CMakeFiles/fiery_post_process.dir/flags.make
CMakeFiles/fiery_post_process.dir/fiery_process.cpp.o: ../fiery_process.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kismet/cppProjects/fiery_post_process/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fiery_post_process.dir/fiery_process.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fiery_post_process.dir/fiery_process.cpp.o -c /home/kismet/cppProjects/fiery_post_process/fiery_process.cpp

CMakeFiles/fiery_post_process.dir/fiery_process.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fiery_post_process.dir/fiery_process.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kismet/cppProjects/fiery_post_process/fiery_process.cpp > CMakeFiles/fiery_post_process.dir/fiery_process.cpp.i

CMakeFiles/fiery_post_process.dir/fiery_process.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fiery_post_process.dir/fiery_process.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kismet/cppProjects/fiery_post_process/fiery_process.cpp -o CMakeFiles/fiery_post_process.dir/fiery_process.cpp.s

CMakeFiles/fiery_post_process.dir/main.cpp.o: CMakeFiles/fiery_post_process.dir/flags.make
CMakeFiles/fiery_post_process.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kismet/cppProjects/fiery_post_process/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/fiery_post_process.dir/main.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/fiery_post_process.dir/main.cpp.o -c /home/kismet/cppProjects/fiery_post_process/main.cpp

CMakeFiles/fiery_post_process.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fiery_post_process.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kismet/cppProjects/fiery_post_process/main.cpp > CMakeFiles/fiery_post_process.dir/main.cpp.i

CMakeFiles/fiery_post_process.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fiery_post_process.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kismet/cppProjects/fiery_post_process/main.cpp -o CMakeFiles/fiery_post_process.dir/main.cpp.s

# Object files for target fiery_post_process
fiery_post_process_OBJECTS = \
"CMakeFiles/fiery_post_process.dir/Hungarian.cpp.o" \
"CMakeFiles/fiery_post_process.dir/fiery_process.cpp.o" \
"CMakeFiles/fiery_post_process.dir/main.cpp.o"

# External object files for target fiery_post_process
fiery_post_process_EXTERNAL_OBJECTS =

fiery_post_process: CMakeFiles/fiery_post_process.dir/Hungarian.cpp.o
fiery_post_process: CMakeFiles/fiery_post_process.dir/fiery_process.cpp.o
fiery_post_process: CMakeFiles/fiery_post_process.dir/main.cpp.o
fiery_post_process: CMakeFiles/fiery_post_process.dir/build.make
fiery_post_process: /home/kismet/libtorch/libtorch/lib/libtorch.so
fiery_post_process: /home/kismet/libtorch/libtorch/lib/libc10.so
fiery_post_process: /home/kismet/libtorch/libtorch/lib/libkineto.a
fiery_post_process: /home/kismet/libtorch/libtorch/lib/libc10.so
fiery_post_process: CMakeFiles/fiery_post_process.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kismet/cppProjects/fiery_post_process/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable fiery_post_process"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/fiery_post_process.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fiery_post_process.dir/build: fiery_post_process

.PHONY : CMakeFiles/fiery_post_process.dir/build

CMakeFiles/fiery_post_process.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/fiery_post_process.dir/cmake_clean.cmake
.PHONY : CMakeFiles/fiery_post_process.dir/clean

CMakeFiles/fiery_post_process.dir/depend:
	cd /home/kismet/cppProjects/fiery_post_process/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kismet/cppProjects/fiery_post_process /home/kismet/cppProjects/fiery_post_process /home/kismet/cppProjects/fiery_post_process/build /home/kismet/cppProjects/fiery_post_process/build /home/kismet/cppProjects/fiery_post_process/build/CMakeFiles/fiery_post_process.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fiery_post_process.dir/depend

