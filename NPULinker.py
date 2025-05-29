import subprocess
import sys
import re

if len(sys.argv) < 2:
    print("Usage: python NPULinker.py [your elf file]")

# 得到二进制文件的数组变量名
bin_id = str(sys.argv[1])
bin_id = re.sub(r'[^a-zA-Z0-9]', '_', bin_id)
if bin_id and bin_id[0].isdigit():
    bin_id = '_' + bin_id
bin_len_id = bin_id + '_len'

# ---- Step1: 获取二进制数组头文件

with open('out_bin.gen.h', 'w') as f:
    subprocess.Popen(['xxd', '-i', str(sys.argv[1])], stdout=f, text=True).wait()
result = subprocess.run(['grep', bin_len_id, 'out_bin.gen.h'], stdout=subprocess.PIPE, text=True)
print(result.stdout)
bin_len = result.stdout.split(' ')[-1][:-2]


# ---- Step2: 抓取Kernel函数信息 -----------
# objdump -t a.elf
objdump = subprocess.Popen(['objdump', '-t', str(sys.argv[1])], stdout=subprocess.PIPE, text=True)
# grep '.npu_kernel'
grep = subprocess.Popen(['grep', '-E', '.npu_kernel|entry_args'], stdin=objdump.stdout, stdout=subprocess.PIPE, text=True)
# awk '{print $6, $1}'
awk_kernel = subprocess.Popen(['awk', "{print $6, $1}"], stdin=grep.stdout, stdout=subprocess.PIPE, text=True)
# 输出
output, _ = awk_kernel.communicate()

# ---- Step3: 输出 -------------------------
kernel_count = 0
args_ptr = ''

out_desc_table = 'static struct NPUKernelDesc kernel_desc_table[128] = {\n'
# 获取kernel名称和地址
for kernel in output.split('\n')[:-1]:
    [name, entry] = kernel.split(' ')
    entry = entry.lstrip('0') or '0'
    if name == 'entry_args':
        args_ptr = entry
        print(f'entry_args Structure address: 0x{args_ptr}')
    else:
        print(f"Kernel Name: {name}, Kernel Entry: 0x{entry}")
        out_desc_table += f'    {{ .entry=0x{entry}, .name="{name}" }},\n'
        kernel_count += 1
out_desc_table += '    { .entry=0x0, .name=NULL }\n};\n'

out_set_desc = 'static struct NPUDeviceKernelSetDesc kernel_set_desc = {\n' + \
               '    .magic = 0,\n' + \
               '    .flags = 0,\n' + \
               '    .version = 0,\n' + \
               '    .reserve1 = 0,\n' + \
               '    .load_start = 0x80000000,\n' + \
              f'    .load_size = {bin_len},\n' + \
              f'    .kernel_count = {kernel_count},\n' + \
               '    .general_entry = 0x80000000,\n' + \
              f'    .args_ptr = 0x{args_ptr},\n' + \
              f'    .data = {bin_id},\n' + \
               '    .kernels = kernel_desc_table\n};'


with open('out_kernel_desc.gen.h', 'w') as f:
    f.write(out_desc_table)
    f.write(out_set_desc)