# Abstract

레이텍에 대해 정리한다.

# Materials

* [LaTeX tutorial](https://www.maths.tcd.ie/~dwilkins/LaTeXPrimer/)
  * 킹왕짱 튜토리얼
* [Learn LaTeX in 30 minutes](https://ko.sharelatex.com/learn/Learn_LaTeX_in_30_minutes)
* [beginngers tutorial](https://ko.sharelatex.com/blog/latex-guides/beginners-tutorial.html)

# References

* [introduction to LaTeX](https://tobi.oetiker.ch/lshort/lshort.pdf)
  * 약 200페이지의 레퍼런스
* [detexify](http://detexify.kirelabs.org/classify.html)
  * 원하는 심볼을 그리면 코드를 알려준다.
* [LaTeX symbols](http://artofproblemsolving.com/wiki/index.php/LaTeX:Symbols)
* [sharelatex](https://ko.sharelatex.com)
* [overleaf](https://www.overleaf.com/)

# Tips

* 한글 사용하기

```
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

* TikZ를 이용하여 image표현하기
  * [TikZ package](https://ko.sharelatex.com/learn/TikZ_package)
  * [TikZ and PGF examples](http://www.texample.net/tikz/examples/all/)
  * [A very minimal introduction to TikZ](http://cremeronline.com/LaTeX/minimaltikz.pdf)
  * [Generating TikZ Code from GeoGebra for LaTeX Documents and Beamer Presentations](https://www.sharelatex.com/blog/2013/08/28/tikz-series-pt2.html)
    * GeoGebra라는 프로그램으로 TikZ code를 만들어 낼 수 있다.
  * [TikZ @ wikibooks](https://en.wikibooks.org/wiki/LaTeX/PGF/TikZ)

