# Abstract

**epoll** is a Linux kernel system call for a scalable I/O event notification mechanism.

# Materials

* [epoll @ wikipedia](https://en.wikipedia.org/wiki/Epoll)
* [Epoll의 기초 개념 및 사용 방법](https://rammuking.tistory.com/entry/Epoll%EC%9D%98-%EA%B8%B0%EC%B4%88-%EA%B0%9C%EB%85%90-%EB%B0%8F-%EC%82%AC%EC%9A%A9-%EB%B0%A9%EB%B2%95)
* [epoll을 사용한비동기 프로그래밍](https://jacking75.github.io/choiheungbae/%EB%AC%B8%EC%84%9C/epoll%EC%9D%84%20%EC%82%AC%EC%9A%A9%ED%95%9C%20%EB%B9%84%EB%8F%99%EA%B8%B0%20%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D.pdf)

# Basic

## API

```c
// Creates an epoll object and returns its file descriptor. 
// The flags parameter allows epoll behavior to be modified. 
// It has only one valid value, EPOLL_CLOEXEC. 
// epoll_create() is an older variant of epoll_create1() and 
// is deprecated as of Linux kernel version 2.6.27 and glibc version 2.9.
int epoll_create1(int flags);

// Controls (configures) which file descriptors are watched by this object, 
// and for which events. op can be ADD, MODIFY or DELETE.
int epoll_ctl(int epfd, int op, int fd, struct epoll_event *event);

// Waits for any of the events registered for with epoll_ctl, 
// until at least one occurs or the timeout elapses. 
// Returns the occurred events in events, up to maxevents at once.
int epoll_wait(int epfd, struct epoll_event *events, int maxevents, int timeout);
```

## Trigerring Modes

epoll provides both **edge-triggered** and **level-triggered** modes.

In case of **level-triggered**, **epoll_wait** will return as long as there are data to be read in buffer.

In case of **edge-triggered**, **epoll_wait** will return just once when the data came to the buffer.

## Examples

* [epoll echo server example @ github](https://github.com/onestraw/epoll-example)
