# Fluid Simulation in GLFW

This project uses `CMake` and `Mingw-w64`

---

---

NOTE: this is a work in progress and everything is not fast

Big thanks to Sebastian Lague - https://github.com/seblague for the fluid simulation lesson
please go visit his channel at - https://www.youtube.com/@SebastianLague

---

## Equations

The `Navier-Strokes` equations (simplified)

-   Equation 1
    $$ \textcolor{orange} a = \frac{- \nabla \textcolor{lightblue} P}{\textcolor{lime} \rho} + \frac{\textcolor{lime} {F\_{external}}}{\textcolor{lime} \rho} + \textcolor{lime} \nu \nabla^2 \textcolor{orange} U $$
-   Equation 1 simplified:
    $$ \textcolor{orange} a = \frac{- \nabla \textcolor{lightblue} P}{\textcolor{lime} \rho} $$
-   Equation 2
    $$ \nabla \cdot \textcolor{orange} U = \textcolor{lime} 0 $$

---

## CMake Configs

`CMakeLists.txt`

```cmake
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
```

`CMakePresets.json`

```json
{
    "version": 8,
    "configurePresets": [
        {
            "name": "GCC 15.2.0 x86_64-w64-mingw32",
            "displayName": "GCC 15.2.0 x86_64-w64-mingw32",
            "description": "Using compilers: C = C:\\mingw64\\bin\\gcc.exe, CXX = C:\\mingw64\\bin\\g++.exe",
            "generator": "MinGW Makefiles",
            "binaryDir": "${sourceDir}/out/build/${presetName}",
            "cacheVariables": {
                "CMAKE_INSTALL_PREFIX": "${sourceDir}/out/install/${presetName}",
                "CMAKE_C_COMPILER": "C:/mingw64/bin/gcc.exe",
                "CMAKE_CXX_COMPILER": "C:/mingw64/bin/g++.exe",
                "CMAKE_BUILD_TYPE": "Debug"
            }
        }
    ]
}
```
