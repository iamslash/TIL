# Abstract

`pgf` 와 `tickz` 는 매크로 패키지이다. `latex` 으로 이미지를 저작하기 위해서 주로 `pgf` 를 backend 로 `tickz` 를 frontend 로 사용한다.

# Materials

  * [A very minimal introduction to TikZ](http://cremeronline.com/LaTeX/minimaltikz.pdf)
  * [TikZ @ wikibooks](https://en.wikibooks.org/wiki/LaTeX/PGF/TikZ)
    * 쉽고 다양한 예제
  * [TikZ package](https://ko.sharelatex.com/learn/TikZ_package)
    * 간단한 설명
  * [TikZ and PGF examples](http://www.texample.net/tikz/examples/all/)
    * 무수한 예제
  * [Generating TikZ Code from GeoGebra for LaTeX Documents and Beamer Presentations](https://www.sharelatex.com/blog/2013/08/28/tikz-series-pt2.html)
    * GeoGebra라는 프로그램으로 TikZ code를 만들어 낼 수 있다.

# Examples

## Basic Structure

```latex
\documentclass[tikz,border=10pt]{article}
\usepackage{tikz}
\begin{document}

\begin{tikzpicture}
code 
\end{tikzpicture}

\end{document}
```