# Abstract

* [Subroutine vs Coroutine - What's the difference?](https://wikidiff.com/coroutine/subroutine)

-----

coroutine 은 여러 thread 가 함께 실행할 수 있는 routine 곧 function 이다. 하나의 thread 가 실행하다가 suspend 되면 다른 thread 가 그 지점에서 resume 할 수 있다.

subroutine 은 coroutine 의 범주에 속한다. 하나의 thread 가 실행할 수 있는 routine 이다.

generator 는 coroutine 의 범주에 속한다. 여러 thread 가 하나의 input 을 가지고 여러개의 output 을 만들어낸다.
