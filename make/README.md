# Abstract

make에 대해 적느다.

# Intro

Makefile의 규칙은 보통 다음과 같다.

```make
target … : prerequisites …
        recipe
        …
        …
```

# References

* [make manual @ gnu](https://www.gnu.org/software/make/manual/make.html#Automatic-Variables)

# Usages

## Automatic Variables

## .PHONY

.PHONY의 prerequisites에 등록된 target들은 해당 target이름과 같은
file이 존재해도 동작 한다.

```
clean :
    rm -rf *.bin
    
.PHONY : clean
```

# Samples
