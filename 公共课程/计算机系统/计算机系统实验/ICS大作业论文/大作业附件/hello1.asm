
hello：     文件格式 elf64-x86-64


Disassembly of section .init:

0000000000401000 <_init>:
  401000:	f3 0f 1e fa          	endbr64 
  401004:	48 83 ec 08          	sub    $0x8,%rsp
  401008:	48 8b 05 e9 2f 00 00 	mov    0x2fe9(%rip),%rax        # 403ff8 <__gmon_start__>
  40100f:	48 85 c0             	test   %rax,%rax
  401012:	74 02                	je     401016 <_init+0x16>
  401014:	ff d0                	callq  *%rax
  401016:	48 83 c4 08          	add    $0x8,%rsp
  40101a:	c3                   	retq   

Disassembly of section .plt:

0000000000401020 <.plt>:
  401020:	ff 35 e2 2f 00 00    	pushq  0x2fe2(%rip)        # 404008 <_GLOBAL_OFFSET_TABLE_+0x8>
  401026:	f2 ff 25 e3 2f 00 00 	bnd jmpq *0x2fe3(%rip)        # 404010 <_GLOBAL_OFFSET_TABLE_+0x10>
  40102d:	0f 1f 00             	nopl   (%rax)
  401030:	f3 0f 1e fa          	endbr64 
  401034:	68 00 00 00 00       	pushq  $0x0
  401039:	f2 e9 e1 ff ff ff    	bnd jmpq 401020 <.plt>
  40103f:	90                   	nop
  401040:	f3 0f 1e fa          	endbr64 
  401044:	68 01 00 00 00       	pushq  $0x1
  401049:	f2 e9 d1 ff ff ff    	bnd jmpq 401020 <.plt>
  40104f:	90                   	nop
  401050:	f3 0f 1e fa          	endbr64 
  401054:	68 02 00 00 00       	pushq  $0x2
  401059:	f2 e9 c1 ff ff ff    	bnd jmpq 401020 <.plt>
  40105f:	90                   	nop
  401060:	f3 0f 1e fa          	endbr64 
  401064:	68 03 00 00 00       	pushq  $0x3
  401069:	f2 e9 b1 ff ff ff    	bnd jmpq 401020 <.plt>
  40106f:	90                   	nop
  401070:	f3 0f 1e fa          	endbr64 
  401074:	68 04 00 00 00       	pushq  $0x4
  401079:	f2 e9 a1 ff ff ff    	bnd jmpq 401020 <.plt>
  40107f:	90                   	nop
  401080:	f3 0f 1e fa          	endbr64 
  401084:	68 05 00 00 00       	pushq  $0x5
  401089:	f2 e9 91 ff ff ff    	bnd jmpq 401020 <.plt>
  40108f:	90                   	nop

Disassembly of section .plt.sec:

0000000000401090 <puts@plt>:
  401090:	f3 0f 1e fa          	endbr64 
  401094:	f2 ff 25 7d 2f 00 00 	bnd jmpq *0x2f7d(%rip)        # 404018 <puts@GLIBC_2.2.5>
  40109b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004010a0 <printf@plt>:
  4010a0:	f3 0f 1e fa          	endbr64 
  4010a4:	f2 ff 25 75 2f 00 00 	bnd jmpq *0x2f75(%rip)        # 404020 <printf@GLIBC_2.2.5>
  4010ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004010b0 <getchar@plt>:
  4010b0:	f3 0f 1e fa          	endbr64 
  4010b4:	f2 ff 25 6d 2f 00 00 	bnd jmpq *0x2f6d(%rip)        # 404028 <getchar@GLIBC_2.2.5>
  4010bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004010c0 <atoi@plt>:
  4010c0:	f3 0f 1e fa          	endbr64 
  4010c4:	f2 ff 25 65 2f 00 00 	bnd jmpq *0x2f65(%rip)        # 404030 <atoi@GLIBC_2.2.5>
  4010cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004010d0 <exit@plt>:
  4010d0:	f3 0f 1e fa          	endbr64 
  4010d4:	f2 ff 25 5d 2f 00 00 	bnd jmpq *0x2f5d(%rip)        # 404038 <exit@GLIBC_2.2.5>
  4010db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000004010e0 <sleep@plt>:
  4010e0:	f3 0f 1e fa          	endbr64 
  4010e4:	f2 ff 25 55 2f 00 00 	bnd jmpq *0x2f55(%rip)        # 404040 <sleep@GLIBC_2.2.5>
  4010eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .text:

00000000004010f0 <_start>:
  4010f0:	f3 0f 1e fa          	endbr64 
  4010f4:	31 ed                	xor    %ebp,%ebp
  4010f6:	49 89 d1             	mov    %rdx,%r9
  4010f9:	5e                   	pop    %rsi
  4010fa:	48 89 e2             	mov    %rsp,%rdx
  4010fd:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
  401101:	50                   	push   %rax
  401102:	54                   	push   %rsp
  401103:	49 c7 c0 30 12 40 00 	mov    $0x401230,%r8
  40110a:	48 c7 c1 c0 11 40 00 	mov    $0x4011c0,%rcx
  401111:	48 c7 c7 25 11 40 00 	mov    $0x401125,%rdi
  401118:	ff 15 d2 2e 00 00    	callq  *0x2ed2(%rip)        # 403ff0 <__libc_start_main@GLIBC_2.2.5>
  40111e:	f4                   	hlt    
  40111f:	90                   	nop

0000000000401120 <_dl_relocate_static_pie>:
  401120:	f3 0f 1e fa          	endbr64 
  401124:	c3                   	retq   

0000000000401125 <main>:
  401125:	f3 0f 1e fa          	endbr64 
  401129:	55                   	push   %rbp
  40112a:	48 89 e5             	mov    %rsp,%rbp
  40112d:	48 83 ec 20          	sub    $0x20,%rsp
  401131:	89 7d ec             	mov    %edi,-0x14(%rbp)
  401134:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
  401138:	83 7d ec 04          	cmpl   $0x4,-0x14(%rbp)
  40113c:	74 14                	je     401152 <main+0x2d>
  40113e:	bf 08 20 40 00       	mov    $0x402008,%edi
  401143:	e8 48 ff ff ff       	callq  401090 <puts@plt>
  401148:	bf 01 00 00 00       	mov    $0x1,%edi
  40114d:	e8 7e ff ff ff       	callq  4010d0 <exit@plt>
  401152:	c7 45 fc 00 00 00 00 	movl   $0x0,-0x4(%rbp)
  401159:	eb 46                	jmp    4011a1 <main+0x7c>
  40115b:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40115f:	48 83 c0 10          	add    $0x10,%rax
  401163:	48 8b 10             	mov    (%rax),%rdx
  401166:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  40116a:	48 83 c0 08          	add    $0x8,%rax
  40116e:	48 8b 00             	mov    (%rax),%rax
  401171:	48 89 c6             	mov    %rax,%rsi
  401174:	bf 2e 20 40 00       	mov    $0x40202e,%edi
  401179:	b8 00 00 00 00       	mov    $0x0,%eax
  40117e:	e8 1d ff ff ff       	callq  4010a0 <printf@plt>
  401183:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
  401187:	48 83 c0 18          	add    $0x18,%rax
  40118b:	48 8b 00             	mov    (%rax),%rax
  40118e:	48 89 c7             	mov    %rax,%rdi
  401191:	e8 2a ff ff ff       	callq  4010c0 <atoi@plt>
  401196:	89 c7                	mov    %eax,%edi
  401198:	e8 43 ff ff ff       	callq  4010e0 <sleep@plt>
  40119d:	83 45 fc 01          	addl   $0x1,-0x4(%rbp)
  4011a1:	83 7d fc 07          	cmpl   $0x7,-0x4(%rbp)
  4011a5:	7e b4                	jle    40115b <main+0x36>
  4011a7:	e8 04 ff ff ff       	callq  4010b0 <getchar@plt>
  4011ac:	b8 00 00 00 00       	mov    $0x0,%eax
  4011b1:	c9                   	leaveq 
  4011b2:	c3                   	retq   
  4011b3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
  4011ba:	00 00 00 
  4011bd:	0f 1f 00             	nopl   (%rax)

00000000004011c0 <__libc_csu_init>:
  4011c0:	f3 0f 1e fa          	endbr64 
  4011c4:	41 57                	push   %r15
  4011c6:	4c 8d 3d 83 2c 00 00 	lea    0x2c83(%rip),%r15        # 403e50 <_DYNAMIC>
  4011cd:	41 56                	push   %r14
  4011cf:	49 89 d6             	mov    %rdx,%r14
  4011d2:	41 55                	push   %r13
  4011d4:	49 89 f5             	mov    %rsi,%r13
  4011d7:	41 54                	push   %r12
  4011d9:	41 89 fc             	mov    %edi,%r12d
  4011dc:	55                   	push   %rbp
  4011dd:	48 8d 2d 6c 2c 00 00 	lea    0x2c6c(%rip),%rbp        # 403e50 <_DYNAMIC>
  4011e4:	53                   	push   %rbx
  4011e5:	4c 29 fd             	sub    %r15,%rbp
  4011e8:	48 83 ec 08          	sub    $0x8,%rsp
  4011ec:	e8 0f fe ff ff       	callq  401000 <_init>
  4011f1:	48 c1 fd 03          	sar    $0x3,%rbp
  4011f5:	74 1f                	je     401216 <__libc_csu_init+0x56>
  4011f7:	31 db                	xor    %ebx,%ebx
  4011f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  401200:	4c 89 f2             	mov    %r14,%rdx
  401203:	4c 89 ee             	mov    %r13,%rsi
  401206:	44 89 e7             	mov    %r12d,%edi
  401209:	41 ff 14 df          	callq  *(%r15,%rbx,8)
  40120d:	48 83 c3 01          	add    $0x1,%rbx
  401211:	48 39 dd             	cmp    %rbx,%rbp
  401214:	75 ea                	jne    401200 <__libc_csu_init+0x40>
  401216:	48 83 c4 08          	add    $0x8,%rsp
  40121a:	5b                   	pop    %rbx
  40121b:	5d                   	pop    %rbp
  40121c:	41 5c                	pop    %r12
  40121e:	41 5d                	pop    %r13
  401220:	41 5e                	pop    %r14
  401222:	41 5f                	pop    %r15
  401224:	c3                   	retq   
  401225:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
  40122c:	00 00 00 00 

0000000000401230 <__libc_csu_fini>:
  401230:	f3 0f 1e fa          	endbr64 
  401234:	c3                   	retq   

Disassembly of section .fini:

0000000000401238 <_fini>:
  401238:	f3 0f 1e fa          	endbr64 
  40123c:	48 83 ec 08          	sub    $0x8,%rsp
  401240:	48 83 c4 08          	add    $0x8,%rsp
  401244:	c3                   	retq   
