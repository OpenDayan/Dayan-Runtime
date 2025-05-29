# NPU-Driver

调用底层 XDMA 驱动，抽象出 NPU 设备初始化， MMIO 读写， memcpy（DMA）读写功能。

# NPU-Runtime

NPU 运行时库，使用 NPU-Driver 层的抽象，提供 contextInit kernelLaunch memcpy 等基础功能支持。

# NPU-Linker

工具流脚本，功能描述：读入 riscv64 elf，转为 kernelDesc 文件，生成对应 .c 文件。
用法：python NPULinker.py YOUR_ELF_FILE
