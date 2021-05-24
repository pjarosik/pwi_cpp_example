## Prerequisites

- CMake version 3.17 at least,
- conan package manager,
- arrrus package version 0.5.13-dev (make sure arrus/lib64 is set in the Path environment variable)

## How to build the application
Update

```
git clone https://github.com/pjarosik/pwi_cpp_example.git
mkdir build
cd build
conan install ..
cmake ..
cmake --build . --config RelWithDebInfo
```
Then run the built application:

```
cd RelWithDebInfo
./pwi_example.exe
```

