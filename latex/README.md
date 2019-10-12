# Abstract

레이텍에 대해 정리한다.

# Materials

* [A simple guide to LaTeX - Step by Step](https://www.latex-tutorial.com/tutorials/)
  * 간결한 튜토리얼 
* [LaTeX로 논문 쓰기 - 입문편](http://t-robotics.blogspot.com/2016/02/latex.html)
* [Learn LaTeX in 30 minutes](https://ko.sharelatex.com/learn/Learn_LaTeX_in_30_minutes)
* [LaTeX tutorial](https://www.maths.tcd.ie/~dwilkins/LaTeXPrimer/)
  * 킹왕짱 튜토리얼
  
# References

* [A LaTeX template](https://github.com/STOM-Group/LaTeX-Paper-Template)
* [introduction to LaTeX](https://tobi.oetiker.ch/lshort/lshort.pdf)
  * 약 200페이지의 레퍼런스
* [detexify](http://detexify.kirelabs.org/classify.html)
  * 원하는 심볼을 그리면 코드를 알려준다.
* [LaTeX symbols](http://artofproblemsolving.com/wiki/index.php/LaTeX:Symbols)
* [sharelatex](https://ko.sharelatex.com)
* [overleaf](https://www.overleaf.com/)

# Overview

`tex` 는 donul knuth 가 만든 프로그램이다. 미려한 문서를 제작할 수 있다. `tex` 로 제작된 소스는 `dvi` 파일로 컴파일될 수 있고 이것은 `ps` 혹은 `pdf` 파일로 변환될 수 있다. 하나의 소스로 다양한 형식의 문서 파일을 만들어 낼 수 있다. `latex` 는 leslie lamport 가 만들었다. `tex` 를 쉽게 사용하기 위한 매크로들의 집합이다.

`pgf` 와 `tickz` 는 매크로 패키지이다. `latex` 으로 이미지를 저작하기 위해서 주로 `pgf` 를 backend 로 `tickz` 를 frontend 로 사용한다.

# Install

## Install with vscode on windows 10

* [Visual Studio Code에서 LaTeX 쓰기](https://hycszero.tistory.com/75)

# Packages

* [Tikz](/tikz/README.md)

# Tips

* 한글 사용하기

```latex
\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{CJKutf8}
\usepackage[english]{babel}

\begin{document}

\begin{CJK}{UTF8}{mj}
한글이 됩니다.
\end{CJK}

\end{document}
```

* [ko.TeX 없이 한글 LaTeX 문서 만들기](https://gist.github.com/dlimpid/5454229)
  * overleaf 에서 xelatex 로 설정하면 잘 된다.

* TikZ를 이용하여 image표현하기
  * [TikZ package](https://ko.sharelatex.com/learn/TikZ_package)
  * [TikZ and PGF examples](http://www.texample.net/tikz/examples/all/)
  * [A very minimal introduction to TikZ](http://cremeronline.com/LaTeX/minimaltikz.pdf)
  * [Generating TikZ Code from GeoGebra for LaTeX Documents and Beamer Presentations](https://www.sharelatex.com/blog/2013/08/28/tikz-series-pt2.html)
    * GeoGebra라는 프로그램으로 TikZ code를 만들어 낼 수 있다.
  * [TikZ @ wikibooks](https://en.wikibooks.org/wiki/LaTeX/PGF/TikZ)