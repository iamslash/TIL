# Abstract

gnu assembler (gas) 에 대해 정리한다.

assembly language 는 intel syntax, AT&T syntax 와 같이 두가지 문법이 있다.
gnu assembler 는 AT&T syntax 를 따른다.

# Materials

* [linux assembly code @ kldp](http://doc.kldp.org/KoreanDoc/html/Assembly_Example-KLDP/Assembly_Example-KLDP.html)

# Basic Usages

## Registers

* general register
  * %eax (%ax, %ah, %al), %ebx (%bx, %bh, %bl), %ecx (%cx, %ch, %cl), %edx (%dx, %dh, %dl), %esi (%si), %edi (%di), %ebp (%bp), %esp (%sp) : 32-bit, 16-bit, 8-bit 레지스터로 사용가능
* section register
  * %cs, %ds, %es, %fs, %gs, %ss
* processor control register
  * %cr0, %cr1, %cr3
* debug register
  * %db0, %db1, %db2, %db3, %db6, %db7
* test register 
  * %tr6, %tr7
* floating point register stack
  * %st => %st(0), %st(1), %st(2), %st(3), %st(4), %st(5), %st(6), %st(7)