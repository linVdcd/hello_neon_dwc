cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# build cpufeatures as a static lib
add_library(cpufeatures STATIC
            ${ANDROID_NDK}/sources/android/cpufeatures/cpu-features.c)

# build app's shared lib

# set up neon build flag for file using intrinsics
# name: helloneon-intrinsics.c (It is named EXACTLY as this on disk,
#                              just like a normal source file)
# then set up neon flag for neon files
# [This example only build for armeabi-v7a, x86 could be done the same way]
#
#

  # make a list of neon files and add neon compiling flags to them
set(neon_SRCS dw.h types.h interface.cpp interface.h)

set_property(SOURCE ${neon_SRCS}
               APPEND_STRING PROPERTY COMPILE_FLAGS " -mfpu=neon")
add_definitions("-DHAVE_NEON=1")




add_library(hello-neon SHARED
            helloneon.c
            ${neon_SRCS})
target_include_directories(hello-neon PRIVATE
                           ${ANDROID_NDK}/sources/android/cpufeatures)

target_link_libraries(hello-neon
                      android
                      cpufeatures
                      log)

