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
extern "C" API_SYMBOL char* IDBS(int* blocks, int* orders);

// 扩展接口：方块数量[7]、放置顺序[7]、抓取坐标[7][5][2]=70、
// 托盘格点坐标[14][10][2]=280、初始拍摄姿态[3] 全部由调用方传入
extern "C" API_SYMBOL char* IDBS_Config(const int* blocks, const int* orders,
                                        const double* block_coord,    // [7][5][2] 共70个
                                        const double* grid_coord,     // [14][10][2] 共280个
                                        const double* shooting_pose); // [3]
