# Compile Caffe

现在的环境纯粹是为了编译caffe而搭建，项目部署时不需要

`cmake-3.9.2 + vs2015 + Anaconda3-2.5.0 (python3.5) + CUDA 8.0 + cuDNN 5`

1. 安装[conda](https://repo.anaconda.com/archive/) 
    - Anaconda3-2.5.0-Windows-x86_64.exe
    - 得到系统default python (3.5)
2. 安装[CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-toolkit-archive)
    - Windows下多版本的CUDA可以共存
        - 不需要切换，只要环境变量PATH中有相应的CUDA路径即可，无需手动切换了
        - 可以使用任何一个版本，只要在环境变量中有对应的CUDA路径即可
3. 安装[cuDNN 5](https://developer.nvidia.com/rdp/cudnn-archive)
    - cuDNN v5 Library for Windows 10
    - 把cuDNN下的lib,include,lib文件夹拷贝到CUDA的安装路径: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
4. 添加系统Path
    ```markdown
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\CUPTI\lib64
    C:\Program Files\NVIDIA Corporation\NVSMI
    C:\Program Files\NVIDIA Corporation\NVIDIA NvDLISR
    C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common
    ```
5. 添加系统环境变量
    ```markdown
    CUDA_PATH_V8_0 = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
    ```
6. 安装[Cmake](https://cmake.org/files/v3.9/cmake-3.9.2-win64-x64.msi)
    - 版本要>=3.4 | 所有选项保持默认即可，一直点击下一步，直到安装完成
    - 添加系统Path: C:\Program Files\CMake\bin
7. 安装[VS2015 community](https://my.visualstudio.com/Downloads?q=visual%20studio%202015&wt.mc_id=o~msft~vscom~older-downloads)
    ```markdown
    - 安装时
        - vs2013=Microsoft Visual Studio 12.0 
        - vs2015=Microsoft Visual Studio 14.0
        - 两者可以安装在一台电脑上并不冲突，可以同时安装，但必须先安装低版本（vs2013）再安装高版本（vs2015），可以兼容
    - 安装时勾选
        - Technically only the VS C/C++ compiler is required (cl.exe)
        - 通用Windows平台开发（包括其子选项C++通用Windows平台工具）
        - 使用C++的桌面开发
    - 安装地址要添加到系统环境
        - vs2013: C:\Program Files (x86)\Microsoft Visual Studio 12.0
        - vs2015: C:\Program Files (x86)\Microsoft Visual Studio 14.0
    ```
    - 若需要安装VS2013 community，[点击此处](https://my.visualstudio.com/Downloads?q=visual%20studio%202013&wt.mc_id=o~msft~vscom~older-downloads)，则Python只能用2.7版本的
8. 编译caffe
    - 介绍
        ```markdown
        caffe是用C++语言编写的深度学习框架，作者是伯克利大学的博士贾扬清，caffe目前主要用于深度学习下的图像处理方面，也就是支持卷积神经网络CNN
        ```
    - 下载caffe-windows源码
        ```bash
        $git clone https://github.com/BVLC/caffe.git
        $cd caffe
        $git checkout windows
        ```
    - 下载Ninjia (用Nijia编译才做这一步，用VS编译不需要这一步)
        ```bash
        $cd xx\caffe
        $git clone git://github.com/ninja-build/ninja.git
        $cd ninja
        # 打开VS2015 x64 Native Tools Command Prompt
        # 将python.exe(3.5版本的)添加到Path
        $python ./configure.py --bootstrap
        # 编译完成(出现一些东西)，将xx\caffe\ninja添加到Path
        ```
    - 修改caffe\scripts\build_win.cmd文件
        - 参考本目录下可正常编译的build_win.cmd: 
            - [cpu-comiple](): 之后detect虚拟环境直接使用cpu运行程序 (tensorflow==1.7.0)
            - [gpu-comiple](): 之后detect虚拟环境配合有CUDA和cuDNN的环境使用 (tensorflow-gpu==1.7.0)
                - 记得选择对应的CUDA_ARCH_NAME的gpu architecture
        - 根据自己的编译器(VS or Nijia)、python版本选择修改
        ```markdown
        - WITH_NINJA=1      | 表示用Ninja编译（或者直接利用vs2015 直接编译也可以，不用安装Ninja）
        - CPU_ONLY=1        | 沒有GPU，或者就是想编译CPU版本
        - CPU_ONLY=0        | 有GPU，想编译GPU版本
            - 在cmake -G"!CMAKE_GENERATOR!"里添加CUDA的路径: 
                - -DCUDNN_ROOT="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0" ^
        - CUDA_ARCH_NAME=Pascal | Fermi  Kepler  Maxwell  Pascal  All (这些都是NVIDIA GPU不同的架构)
            - 对于我的 DELL Desktop: NIVIDA GeForce GTX 1050 就是采用的Pascal架构
            - 对于Azure VM 和 Windows Server： NVIDIA Tesla V100 采用的是Volta架构
        - PYTHON_VERSION=3  | 表示python3.x版本接口
            - 相应地: set CONDA_ROOT=C:\Anaconda3 (C:\Anaconda3\python.exe是Python 3.5.1 :: Anaconda 2.5.0 (64-bit))
        - CONDA_ROOT=上面python3.5的路径
        - BUILD_PYTHON=1
        - RUN_INSTALL=1
        ```
    - C:\Users\PC\.caffe\dependencies\libraries_v140_x64_py35_1.1.0\libraries\include\boost-1_61\boost\config\compiler\nvcc.hpp 删除最后三行
    - build之前的准备
        ```markdown
        - python.exe的路径一定只能有python3.5的，如果有conda的python.exe(非3.5版本的)先删除，build完再恢复
        - 否则会报错: Found PythonInterp: C:/ProgramData/Anaconda3/python.exe (found suitable version "3.8.5", minimum required is "2.7")
        ```
    - 编译caffe
        - 在caffe\scripts下新建build文件夹，并清空里面的所有文件
        - 以“管理员权限”打开VS2015 x64 Native Tools Command Prompt
            ```bash
            $cd scripts
            $build_win.cmd
            # 每次运行build_win.cmd之前都要先清空build文件夹
            # VS2015 build结果: 9037 个警告 | 0 个错误 | 已用时间 00:22:02.25  √
            ```
9. 添加Python接口到detect.py运行的虚拟环境
    - 将caffe\python\caffe拷贝到虚拟环境的\Lib\site-packages
    ```bash
    $python
    $import caffe
    ```
    
## Compile Error

1. Unsupported gpu architecture 'compute_90'
    ```markdown
    CustomBuild:
    Building NVCC (Device) object src/caffe/CMakeFiles/cuda_compile_1.dir/layers/Release/cuda_compile_1_generated_absval_
    layer.cu.obj
    nvcc fatal   : Unsupported gpu architecture 'compute_90'
    CMake Error at cuda_compile_1_generated_absval_layer.cu.obj.Release.cmake:222 (message):
        Error generating
        C:/Path/caffe/scripts/build/src/caffe/CMakeFiles/cuda_compile_1.dir/layers/Release/cuda_compile_1_generated_absval_
    layer.cu.obj


    C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V140\Microsoft.CppCommon.targets(171,5): error MSB6006: “cmd.exe”已 退出，
    代码为 1。 [C:\Path\caffe\scripts\build\src\caffe\caffe.vcxproj]
    已完成生成项目“C:\Path\caffe\scripts\build\src\caffe\caffe.vcxproj”(默认目标)的操作 - 失败。

    已完成生成项目“C:\Path\caffe\scripts\build\ALL_BUILD.vcxproj”(默认目标)的操作 - 失败。
    - 解决方法: C:\Users\PC\.caffe\dependencies\libraries_v140_x64_py35_1.1.0\libraries\include\boost-1_61\boost\config\compiler\nvcc.hpp 删除最后三行
    ```