set(projName FluidSim)

cmake_minimum_required(VERSION 3.10.0)
project(${projName} VERSION 0.1.0 LANGUAGES C CXX)

# === Library directories ===
set(GLFW_DIR libraries/GLFW-x64)
set(GLAD_DIR libraries/GLAD-x86_64)
set(GLM_DIR  libraries/GLM)

# === Include paths ===
include_directories(
    ${GLFW_DIR}/include
    ${GLAD_DIR}/include
    ${GLM_DIR}
)

# === Link GLFW lib folder ===
link_directories(${GLFW_DIR}/lib-mingw-w64)

# === Add GLAD ===
add_library(glad STATIC "${GLAD_DIR}/src/glad.c")

# === Executable ===
add_executable(${PROJECT_NAME} src/main.cpp)

# === Link everything ===
target_link_libraries(${PROJECT_NAME}
    glad
    glfw3
    opengl32
)