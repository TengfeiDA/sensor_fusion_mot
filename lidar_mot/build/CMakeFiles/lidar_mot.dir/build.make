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
CMAKE_SOURCE_DIR = /home/tengfeida/sensor_fusion_mot/lidar_mot

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tengfeida/sensor_fusion_mot/lidar_mot/build

# Include any dependencies generated for this target.
include CMakeFiles/lidar_mot.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/lidar_mot.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/lidar_mot.dir/flags.make

CMakeFiles/lidar_mot.dir/main.cc.o: CMakeFiles/lidar_mot.dir/flags.make
CMakeFiles/lidar_mot.dir/main.cc.o: ../main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tengfeida/sensor_fusion_mot/lidar_mot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/lidar_mot.dir/main.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lidar_mot.dir/main.cc.o -c /home/tengfeida/sensor_fusion_mot/lidar_mot/main.cc

CMakeFiles/lidar_mot.dir/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lidar_mot.dir/main.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tengfeida/sensor_fusion_mot/lidar_mot/main.cc > CMakeFiles/lidar_mot.dir/main.cc.i

CMakeFiles/lidar_mot.dir/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lidar_mot.dir/main.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tengfeida/sensor_fusion_mot/lidar_mot/main.cc -o CMakeFiles/lidar_mot.dir/main.cc.s

CMakeFiles/lidar_mot.dir/track.cc.o: CMakeFiles/lidar_mot.dir/flags.make
CMakeFiles/lidar_mot.dir/track.cc.o: ../track.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tengfeida/sensor_fusion_mot/lidar_mot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/lidar_mot.dir/track.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lidar_mot.dir/track.cc.o -c /home/tengfeida/sensor_fusion_mot/lidar_mot/track.cc

CMakeFiles/lidar_mot.dir/track.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lidar_mot.dir/track.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tengfeida/sensor_fusion_mot/lidar_mot/track.cc > CMakeFiles/lidar_mot.dir/track.cc.i

CMakeFiles/lidar_mot.dir/track.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lidar_mot.dir/track.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tengfeida/sensor_fusion_mot/lidar_mot/track.cc -o CMakeFiles/lidar_mot.dir/track.cc.s

CMakeFiles/lidar_mot.dir/tracker.cc.o: CMakeFiles/lidar_mot.dir/flags.make
CMakeFiles/lidar_mot.dir/tracker.cc.o: ../tracker.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tengfeida/sensor_fusion_mot/lidar_mot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/lidar_mot.dir/tracker.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lidar_mot.dir/tracker.cc.o -c /home/tengfeida/sensor_fusion_mot/lidar_mot/tracker.cc

CMakeFiles/lidar_mot.dir/tracker.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lidar_mot.dir/tracker.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tengfeida/sensor_fusion_mot/lidar_mot/tracker.cc > CMakeFiles/lidar_mot.dir/tracker.cc.i

CMakeFiles/lidar_mot.dir/tracker.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lidar_mot.dir/tracker.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tengfeida/sensor_fusion_mot/lidar_mot/tracker.cc -o CMakeFiles/lidar_mot.dir/tracker.cc.s

CMakeFiles/lidar_mot.dir/visualizer.cc.o: CMakeFiles/lidar_mot.dir/flags.make
CMakeFiles/lidar_mot.dir/visualizer.cc.o: ../visualizer.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tengfeida/sensor_fusion_mot/lidar_mot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/lidar_mot.dir/visualizer.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/lidar_mot.dir/visualizer.cc.o -c /home/tengfeida/sensor_fusion_mot/lidar_mot/visualizer.cc

CMakeFiles/lidar_mot.dir/visualizer.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/lidar_mot.dir/visualizer.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/tengfeida/sensor_fusion_mot/lidar_mot/visualizer.cc > CMakeFiles/lidar_mot.dir/visualizer.cc.i

CMakeFiles/lidar_mot.dir/visualizer.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/lidar_mot.dir/visualizer.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/tengfeida/sensor_fusion_mot/lidar_mot/visualizer.cc -o CMakeFiles/lidar_mot.dir/visualizer.cc.s

# Object files for target lidar_mot
lidar_mot_OBJECTS = \
"CMakeFiles/lidar_mot.dir/main.cc.o" \
"CMakeFiles/lidar_mot.dir/track.cc.o" \
"CMakeFiles/lidar_mot.dir/tracker.cc.o" \
"CMakeFiles/lidar_mot.dir/visualizer.cc.o"

# External object files for target lidar_mot
lidar_mot_EXTERNAL_OBJECTS =

lidar_mot: CMakeFiles/lidar_mot.dir/main.cc.o
lidar_mot: CMakeFiles/lidar_mot.dir/track.cc.o
lidar_mot: CMakeFiles/lidar_mot.dir/tracker.cc.o
lidar_mot: CMakeFiles/lidar_mot.dir/visualizer.cc.o
lidar_mot: CMakeFiles/lidar_mot.dir/build.make
lidar_mot: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
lidar_mot: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
lidar_mot: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
lidar_mot: CMakeFiles/lidar_mot.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tengfeida/sensor_fusion_mot/lidar_mot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable lidar_mot"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/lidar_mot.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/lidar_mot.dir/build: lidar_mot

.PHONY : CMakeFiles/lidar_mot.dir/build

CMakeFiles/lidar_mot.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/lidar_mot.dir/cmake_clean.cmake
.PHONY : CMakeFiles/lidar_mot.dir/clean

CMakeFiles/lidar_mot.dir/depend:
	cd /home/tengfeida/sensor_fusion_mot/lidar_mot/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tengfeida/sensor_fusion_mot/lidar_mot /home/tengfeida/sensor_fusion_mot/lidar_mot /home/tengfeida/sensor_fusion_mot/lidar_mot/build /home/tengfeida/sensor_fusion_mot/lidar_mot/build /home/tengfeida/sensor_fusion_mot/lidar_mot/build/CMakeFiles/lidar_mot.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/lidar_mot.dir/depend

