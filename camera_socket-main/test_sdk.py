#!/usr/bin/env python3
"""
MVS SDK（海康威视相机库）可用性测试脚本

验证内容（无需连接实体相机）：
1. libMvCameraControl.so 能否被 ctypes 加载
2. MvCameraControl_class 模块能否正常 import（所有 header 依赖齐全）
3. MV_CC_EnumDevices 能否调用（无设备时返回 0、nDeviceNum=0，证明库层可用）
4. 打印 SDK 路径、版本信息、架构、环境变量状态

用法（在 camera conda 环境下、项目根目录）:
    python test_sdk.py
"""
import ctypes
import os
import platform
import sys

ARCH = platform.machine()
SDK_DEFAULT_ROOT = "/opt/MVS/lib"
ARCH_MAP = {
    "aarch64": "/aarch64",
    "x86_64": "/64",
    "i386": "/32", "i686": "/32",
    "armhf": "/armhf", "armv6l": "/armhf", "armv7l": "/armhf",
}


def sep(t="="):
    print(t * 60)


def main():
    sep()
    print("MVS SDK 可用性测试")
    sep()
    print(f"架构            : {ARCH}")
    print(f"MVCAM_COMMON_RUNENV = '{os.getenv('MVCAM_COMMON_RUNENV', '<未设置>')}'")
    print(f"MVCAM_SDK_PATH     = '{os.getenv('MVCAM_SDK_PATH', '<未设置>')}'")

    suffix = ARCH_MAP.get(ARCH, "/64")
    sdk_root = os.getenv("MVCAM_COMMON_RUNENV") or SDK_DEFAULT_ROOT
    so_path = sdk_root + suffix + "/libMvCameraControl.so"
    print(f"预期 .so 路径  : {so_path}")

    # ---- 1. .so 文件是否存在 ----
    sep("-")
    print("[1/4] 检查 .so 文件是否存在 ...")
    if not os.path.exists(so_path):
        print(f"  ❌ 未找到 {so_path}")
        print("     请确认已安装 Linux aarch64 版 MVS SDK（默认 /opt/MVS）。")
        return
    print(f"  ✅ 找到: {so_path}")

    # ---- 2. ctypes 能否加载 ----
    sep("-")
    print("[2/4] ctypes 加载 libMvCameraControl.so ...")
    try:
        lib = ctypes.cdll.LoadLibrary(so_path)
        print(f"  ✅ 加载成功: {lib}")
    except OSError as e:
        print(f"  ❌ 加载失败: {e}")
        print("     可能缺少依赖库。运行 `ldd {}` 检查。".format(so_path))
        return

    # ---- 3. import MvCameraControl_class（依赖所有 header 模块）----
    sep("-")
    print("[3/4] import MvCameraControl_class（含全部 header 依赖）...")
    try:
        import MvCameraControl_class as sdk  # noqa: F401
        from MvCameraControl_class import (  # noqa: F401
            MvCamera, MV_CC_DEVICE_INFO_LIST, MV_GIGE_DEVICE, MV_USB_DEVICE,
        )
        print("  ✅ 模块导入成功（PixelType_header/CameraParams/MvErrorDefine 等依赖齐全）")
    except Exception as e:
        print(f"  ❌ 导入失败: {type(e).__name__}: {e}")
        return

    # ---- 4. 调用 MV_CC_EnumDevices（无设备也应正常返回）----
    sep("-")
    print("[4/4] 调用 MV_CC_EnumDevices 枚举设备（无需实体相机）...")
    try:
        dev_list = MV_CC_DEVICE_INFO_LIST()
        tlayer = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayer, dev_list)
        n = dev_list.nDeviceNum
        print(f"  ✅ 调用成功: 返回码 ret=0x{ret & 0xFFFFFFFF:08X}, 发现设备数 = {n}")
        if ret == 0:
            print("  ✅ 库层完全可用（返回码 0 = 成功）")
        else:
            print(f"  ⚠️ 库可调用但返回非 0（0x{ret & 0xFFFFFFFF:08X}），仍证明库已加载")
        if n == 0:
            print("  ℹ️ 当前无相机连接（预期情况，本次不连接实体相机）")
        else:
            print(f"  🎉 检测到 {n} 台相机！")
    except Exception as e:
        print(f"  ❌ 调用失败: {type(e).__name__}: {e}")
        return

    # ---- 结论 ----
    sep()
    print("结论: ✅ MVS SDK 库可正常加载和调用")
    print("     （当前无实体相机时 EnumDevices 返回 0 设备，属正常；连接相机后即可采集）")
    sep()


if __name__ == "__main__":
    sys.exit(main())
