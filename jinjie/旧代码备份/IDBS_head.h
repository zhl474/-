#pragma once
//跨平台头文件
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>  // Linux 动态加载
#include <unistd.h> // Linux 系统调用
#endif

//跨平台导出宏定义
#if defined(_WIN32)
    #ifdef BUILD_TEST
        #define API_SYMBOL __declspec(dllexport)
    #else
        #define API_SYMBOL __declspec(dllimport)
    #endif
#elif defined(__linux__) || defined(__unix__) || defined(__APPLE__)
    #ifdef BUILD_TEST
        #define API_SYMBOL __attribute__((visibility("default")))
    #else
        #define API_SYMBOL
    #endif
#else
    #define API_SYMBOL
#endif

//函数声明
extern "C" API_SYMBOL char* IDBS(int block1, int block2, int block3, int block4, int block5, int block6, int block7,
                                       int order1, int order2, int order3, int order4, int order5, int order6, int order7);

// V1 版本化接口：在原有类别、角度和中心之外返回四个运行时托盘占用格。
extern "C" API_SYMBOL char* IDBSWithCells(int block1, int block2, int block3, int block4, int block5, int block6, int block7,
                                                int order1, int order2, int order3, int order4, int order5, int order6, int order7);
