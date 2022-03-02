cmake_minimum_required(VERSION 3.2)
project(csc4005_imgui)

set(CMAKE_CXX_STANDARD 17)

find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)

include_directories(include imgui imgui/backends ${SDL2_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIRS})
file(GLOB IMGUI_SRC
        imgui/*.cpp
        imgui/backends/imgui_impl_sdl.cpp
        imgui/backends/imgui_impl_opengl2.cpp
        imgui/misc/freetype/imgui_freetype.cpp
        imgui/misc/fonts/binary_to_compressed_c.cpp
        imgui/misc/cpp/imgui_stdlib.cpp
        )
add_library(imgui STATIC ${IMGUI_SRC})
file(GLOB PUBLIC_SRC src/graphic.cpp)

add_executable(main_1 src/main_1.cpp ${PUBLIC_SRC})
add_executable(main_2 src/main_2.cpp ${PUBLIC_SRC})
add_executable(main_3 src/main_3.cpp ${PUBLIC_SRC})
add_executable(main_4 src/main_4.cpp ${PUBLIC_SRC})

get_filename_component(FONT_PATH imgui/misc/fonts/DroidSans.ttf ABSOLUTE)
target_link_libraries(imgui PRIVATE Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS})

target_link_libraries(main_1 imgui)
target_compile_options(main_1 PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_link_libraries(main_2 imgui)
target_compile_options(main_2 PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(main_3 imgui)
target_compile_options(main_3 PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(main_4 imgui)
target_compile_options(main_4 PRIVATE -Wall -Wextra -Wpedantic)

target_compile_definitions(imgui PUBLIC -DImDrawIdx=unsigned)

target_compile_definitions(main_1 PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(main_2 PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(main_3 PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(main_4 PRIVATE -DFONT_PATH=\"${FONT_PATH}\")