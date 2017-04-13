# llvm install

- [llvm download](http://releases.llvm.org/download.html)

# llvm usage

```bash
clang -emit-llvm -S a.c -o a.ll
opt -O3 -S a.ll -o a.opt.ll
llc -03 a.opt.ll -o a.s
gcc a.s
```
