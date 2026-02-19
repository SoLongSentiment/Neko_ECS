cmake_minimum_required(VERSION 3.16)
project(StringProject VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -march=native  -fno-plt)
add_compile_options(-fno-stack-protector -fomit-frame-pointer -flto=full)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-mllvm -inline-threshold=5000)
    add_link_options(-static) 
    add_link_options(-static-libgcc -static-libstdc++) 
endif()

add_executable(ECS_bench main.cpp)
target_compile_options(ECS_bench PRIVATE -falign-loops=32)