
hello.o：     文件格式 elf64-x86-64


Disassembly of section .text:

0000000000000000 <main>:
   0:	f3 0f 1e fa          	endbr64 
   4:	55                   	push   %rbp
   5:	48 89 e5             	mov    %rsp,%rbp
   8:	48 83 ec 20          	sub    $0x20,%rsp
   c:	89 7d ec             	mov    %edi,-0x14(%rbp)
   f:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  13:	83 7d ec 04          	cmpl   $0x4,-0x14(%rbp)
  17:	74 14                	je     2d <main+0x2d>
  19:	bf 00 00 00 00       	mov    $0x0,%edi
			1a: R_X86_64_32	.rodata
  1e:	e8 00 00 00 00       	callq  23 <main+0x23>
			1f: R_X86_64_PLT32	puts-0x4
  23:	bf 01 00 00 00       	mov    $0x1,%edi
  28:	e8 00 00 00 00       	callq  2d <main+0x2d>
			29: R_X86_64_PLT32	exit-0x4
  2d:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  34:	eb 46                	jmp    7c <main+0x7c>
  36:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  3a:	48 83 c0 10          	add    $0x10,%rax
  3e:	48 8b 10             	mov    (%rax),%rdx
  41:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  45:	48 83 c0 08          	add    $0x8,%rax
  49:	48 8b 00             	mov    (%rax),%rax
  4c:	48 89 c6             	mov    %rax,%rsi
  4f:	bf 00 00 00 00       	mov    $0x0,%edi
			50: R_X86_64_32	.rodata+0x26
  54:	b8 00 00 00 00       	mov    $0x0,%eax
  59:	e8 00 00 00 00       	callq  5e <main+0x5e>
			5a: R_X86_64_PLT32	printf-0x4
  5e:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  62:	48 83 c0 18          	add    $0x18,%rax
  66:	48 8b 00             	mov    (%rax),%rax
  69:	48 89 c7             	mov    %rax,%rdi
  6c:	e8 00 00 00 00       	callq  71 <main+0x71>
			6d: R_X86_64_PLT32	atoi-0x4
  71:	89 c7                	mov    %eax,%edi
  73:	e8 00 00 00 00       	callq  78 <main+0x78>
			74: R_X86_64_PLT32	sleep-0x4
  78:	83 45 fc 01          	addl   $0x1,-0x4(%rbp)
  7c:	83 7d fc 07          	cmpl   $0x7,-0x4(%rbp)
  80:	7e b4                	jle    36 <main+0x36>
  82:	e8 00 00 00 00       	callq  87 <main+0x87>
			83: R_X86_64_PLT32	getchar-0x4
  87:	b8 00 00 00 00       	mov    $0x0,%eax
  8c:	c9                   	leaveq 
  8d:	c3                   	retq   
