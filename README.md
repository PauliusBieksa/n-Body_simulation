# n-Body_simulation
A comparison of simple sequential and parallelised implementations of the n-body problem.

This project requires CUDA runtime libraries.

To compile:
1. Clone this repo.
2. Update submodules (git submodule uptate --init --recursive)
3. Use CMake to build the project
4. Build copy_res project to copy the shaders to your build directory.
5. Enable OMP in OMP_PARALLEL project settings in Visual Studio
(right-click on OMP_PARALLEL in solution explorer -> properties -> expand C/C++ -> Language -> set 'Open MP Support' to yes)
