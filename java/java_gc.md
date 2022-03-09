# Mark and Sweep

* [Java Garbage Collection Basics](https://www.oracle.com/webfolder/technetwork/tutorials/obe/java/gc01/index.html)
  * garbage collector 의 기본 원리에 대해 알 수 있다.

![](https://www.oracle.com/webfolder/technetwork/tutorials/obe/java/gc01/images/gcslides/Slide5.png)

jvm 의 gc 는 크게 `Young Generation, Old Generation, Permanent Generation` 으로 나누어 진다. 영원히 보존해야 되는 것들은 `Permanent Generation` 으로 격리된다. `Young Generation` 에서 `minor collect` 할때 마다 살아남은 녀석들중 나이가 많은 녀석들은 `Old Generation` 으로 격리된다. 

`Young Generation` 은 다시 `eden, S0, S1` 으로 나누어 진다. `eden` 이 꽉 차면 `minor collect` 이벤트가 발생하고 `eden, S0` 혹은 `eden, S1` 의 `unreferenced object` 는 소멸되고 `referenced object` 는 나이가 하나 증가하여 `S1` 혹은 `S0` 으로 옮겨진다. 나이가 많은 녀석들은 `Old Genration` 으로 옮겨진다. `eden, S0` 과 `eden, S1` 이 교대로 사용된다.

# G1GC (Garbage First Garbage Collector)

* [Java HotSpot VM G1GCs @ 기계인간](https://johngrib.github.io/wiki/java-g1gc/)
* [Java G1 GC @ 권남](https://kwonnam.pe.kr/wiki/java/g1gc)
* [일반적인 GC 내용과 G1GC (Garbage-First Garbage Collector) 내용](https://thinkground.studio/%EC%9D%BC%EB%B0%98%EC%A0%81%EC%9D%B8-gc-%EB%82%B4%EC%9A%A9%EA%B3%BC-g1gc-garbage-first-garbage-collector-%EB%82%B4%EC%9A%A9/)

-----


