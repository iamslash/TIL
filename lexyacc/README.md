# Materials

* [Flex and Bison Howto](https://www.joinc.co.kr/w/Site/Development/Env/Yacc)
  * 한글 튜토리얼 초급
* [Lex & Yacc(소개와 예제 중심으로)](http://nlp.kookmin.ac.kr/sskang/lect/compiler/lexyacc/Lex_Yacc.htm)
  * 한글 튜토리얼 중급
* [The Lex & Yacc Page](http://dinosaur.compilertools.net/)
  * 자세한 소개

# References

* [basiccalculator @ github](https://github.com/iamslash/basiccalculator)
  * 간단한 계산기

# lex

다음은 `a.l` 구조이다.

```
definitions 
  # rules에 pattern을 간단하게 하기 위한 선언과 초기 조건
  # C코드 삽입시 %{, }% 기호를 표시하고 사이에 쓸 수 있음
%%
rules  
  # pattern과 action으로 이루어짐
  # C코드는 {, }로 감싸서 표시함
%%
user code 
  # yylex()함수와 사용자가 원하는 C 루틴으로 이루어짐
```

# yacc

다음은 `a.y` 의 구조이다.

```
definitions 
   # C코드 삽입시 %{, }% 기호로 표시하고 사이에 쓸 수 있음
%%
rules 
   # 각 rule은 "LHS: RHS;"와 같은 형식으로 이루어짐
%%
user code
   # lex와의 결합시 yylex()를 이용한다.
```