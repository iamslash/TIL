# llvm install

- [llvm download](http://releases.llvm.org/download.html)

# llvm usage

- a.c

```c
#include <stdio.h>
int main() { printf("hello world\n"); }
```

```bash
clang --help
clang file.c -fsyntax-only (check for correctness)
clang file.c -S -emit-llvm -o - (print out unoptimized llvm code)
clang file.c -S -emit-llvm -o - -O3
clang file.c -S -O3 -o - (output native machine code)
```
