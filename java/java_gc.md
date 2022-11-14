- [Mark and Sweep](#mark-and-sweep)
- [G1GC (Garbage First Garbage Collector)](#g1gc-garbage-first-garbage-collector)

----

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

G1GC 는 Full GC 가 발생하면 Memory 를 OS 에 돌려주는 것 같다.

```cpp
// hotspot/src/share/vm/gc_implementation/g1/g1CollectedHeap.cpp
bool G1CollectedHeap::do_collection(bool explicit_gc,
                                    bool clear_all_soft_refs,
                                    size_t word_size) {
...
      // Resize the heap if necessary.
      resize_if_necessary_after_full_collection(explicit_gc ? 0 : word_size);
...
}                                      

// hotspot/src/share/vm/gc_implementation/g1/g1CollectedHeap.hpp
class G1CollectedHeap : public SharedHeap {
...
  // Resize the heap if necessary after a full collection.  If this is
  // after a collect-for allocation, "word_size" is the allocation size,
  // and will be considered part of the used portion of the heap.
  void resize_if_necessary_after_full_collection(size_t word_size);
...  
}

// hotspot/src/share/vm/gc_implementation/g1/g1CollectedHeap.cpp

```

