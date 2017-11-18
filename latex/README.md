# Abstract

레이텍에 대해 정리한다.

# Materials

* [Learn LaTeX in 30 minutes](https://ko.sharelatex.com/learn/Learn_LaTeX_in_30_minutes)
* [beginngers tutorial](https://ko.sharelatex.com/blog/latex-guides/beginners-tutorial.html)

# References

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
  * [Generating TikZ Code from GeoGebra for LaTeX Documents and Beamer Presentations](https://www.sharelatex.com/blog/2013/08/28/tikz-series-pt2.html)
    * GeoGebra라는 프로그램으로 TikZ code를 만들어 낼 수 있다.

