- [References](#references)
- [Materials](#materials)
- [Operators](#operators)
  - [map](#map)
  - [mapTo](#mapto)
  - [merge](#merge)
  - [mergeAll](#mergeall)
  - [mergeMap](#mergemap)
  - [mergeMapTo](#mergemapto)
- [Advanced](#advanced)
  - [merge vs mergeAll vs mergeMap](#merge-vs-mergeall-vs-mergemap)

---

# References

* [Rx Visualizer](https://rxviz.com/)
  * 킹왕짱 rx 시각화 툴
* [jsfiddle](https://jsfiddle.net/)
  * javascript web-ide

# Materials

* [rx.js @ yalco](https://www.yalco.kr/lectures/rxjs/)
  * [src @ gitlab](https://gitlab.com/yalco/yalco-rxjs-practice-server)
  * [reactive programming @ inflearn](https://www.inflearn.com/course/%EC%96%84%EC%BD%94-%EC%9E%90%EB%B0%94%EC%8A%A4%ED%81%AC%EB%A6%BD%ED%8A%B8-reactivex?inst=3a1f0365#curriculum)
* [learn rx.js](https://www.learnrxjs.io/)
* [rx.js @ udemy](https://matchgroup.udemy.com/course/rxjs-course/learn/lecture/10897630#reviews)
* [reactive how](https://reactive.how/)
  * reactive 시각화

# Operators

## map

## mapTo

## merge

## mergeAll

## mergeMap

## mergeMapTo

# Advanced

## merge vs mergeAll vs mergeMap

* [RxJS: merge() vs. mergeAll() vs. mergeMap()](https://medium.com/@damianczapiewski/rxjs-merge-vs-mergeall-vs-mergemap-7d8f40fc4756)

----

```js
// merge
import { interval, merge } from 'rxjs';
import { take } form 'rxjs/operators';
const intA$ = interval(750).pipe(take(3));
const intB$ = interval(1000).pipe(take(3));
merge(intA$, intB$)
  .subscribe(console.log);
// after 750ms emits 0 from intA$
// after 1s    emits 0 from intB$
// after 1.5s  emits 1 from intA$
// after 2s    emits 1 from intB$
// after 2.25s emits 2 from intA$
// after 3s    emits 3 from intB$

// mergeAll
import { interval } from 'rxjs';
import { map, take } from 'rxjs/operators';
const int$ = interval(1000).pipe(
  take(2),
  map(int => interval(500).pipe(take(3))
);
// map 의 결과는 observable 이다. 두개의 Observable 을 출력한다.
int$.subscribe(console.log);
// Observable
// Observable

import { interval } from 'rxjs';
import { map, take, mergeAll } from 'rxjs/operators';
const int$ = interval(1000).pipe(
  take(2),
  map(int => interval(500).pipe(take(3)),
  mergeAll()
);
// map 에서 Observable 을 리턴한다.
// mergeAll() 을 했으므로 Observable 들을 flatten 해서 합친다.
int$.subscribe(console.log);

// mergeMap
import { interval } from 'rxjs';
import { take, mergeMap } from 'rxjs/operators';
const int$ = interval(1000).pipe(
  take(2),
  mergeMap(int => interval(500).pipe(take(3))
);
// mergeMap = map + mergeAll
int$.subscribe(console.log);
```
