# 简介

信息内容安全本科生实验三

# 使用方法

```bash
make all
./StringMatch -h
Usage: ./StringMatch <pattern_file> <text_file> <method_choice> [length_threshold]
```

# 实验结果

```bash
root@hcss-ecs-3bd4:~/Code/lab-3-StringMatch# ./StringMatch pattern1w.txt text.txt 1
[AC] Initialization_time=299ms
进程 3368718 的虚拟内存使用 (VSize): 191.50 MB
[AC] Search_time=425ms Hit_num=267377
root@hcss-ecs-3bd4:~/Code/lab-3-StringMatch# ./StringMatch pattern2w.txt text.txt 1
[AC] Initialization_time=768ms
进程 3368751 的虚拟内存使用 (VSize): 355.48 MB
[AC] Search_time=678ms Hit_num=451305
root@hcss-ecs-3bd4:~/Code/lab-3-StringMatch# ./StringMatch pattern3w.txt text.txt 1
[AC] Initialization_time=2468ms
进程 3368771 的虚拟内存使用 (VSize): 722.21 MB
[AC] Search_time=831ms Hit_num=467021
root@hcss-ecs-3bd4:~/Code/lab-3-StringMatch# ./StringMatch pattern1w.txt text.txt 2
[WM] Initialization_time=182ms
进程 3368791 的虚拟内存使用 (VSize): 4.30 MB
[WM] Search_time=587ms Hit_num=267377
root@hcss-ecs-3bd4:~/Code/lab-3-StringMatch# ./StringMatch pattern2w.txt text.txt 2
[WM] Initialization_time=977ms
进程 3368821 的虚拟内存使用 (VSize): 5.61 MB
[WM] Search_time=1082ms Hit_num=451305
root@hcss-ecs-3bd4:~/Code/lab-3-StringMatch# ./StringMatch pattern3w.txt text.txt 2
[WM] Initialization_time=2390ms
进程 3368854 的虚拟内存使用 (VSize): 7.15 MB
[WM] Search_time=1606ms Hit_num=467021
root@hcss-ecs-3bd4:~/Code/lab-3-StringMatch# ./StringMatch pattern1w.txt text.txt 3 15
[WM] Initialization_time=19ms
进程 3368892 的虚拟内存使用 (VSize): 3.72 MB
[WM] Search_time=157ms Hit_num=23127
[AC] Initialization_time=100ms
进程 3368892 的虚拟内存使用 (VSize): 72.19 MB
[AC] Search_time=295ms Hit_num=244250
root@hcss-ecs-3bd4:~/Code/lab-3-StringMatch# ./StringMatch pattern2w.txt text.txt 3 15
[WM] Initialization_time=175ms
进程 3368928 的虚拟内存使用 (VSize): 4.75 MB
[WM] Search_time=260ms Hit_num=73386
[AC] Initialization_time=158ms
进程 3368928 的虚拟内存使用 (VSize): 115.65 MB
[AC] Search_time=391ms Hit_num=377919
root@hcss-ecs-3bd4:~/Code/lab-3-StringMatch# ./StringMatch pattern3w.txt text.txt 3 15
[WM] Initialization_time=968ms
进程 3368949 的虚拟内存使用 (VSize): 6.29 MB
[WM] Search_time=380ms Hit_num=89102
[AC] Initialization_time=155ms
进程 3368949 的虚拟内存使用 (VSize): 116.96 MB
[AC] Search_time=379ms Hit_num=377919
```