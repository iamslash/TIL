# Materials

* [Sticky Session과 Session Clustering](https://smjeon.dev/web/sticky-session/)
* [ALB sticky session 에 대한 고찰](https://linuxer.name/2019/10/alb-sticky-session-%EC%97%90-%EB%8C%80%ED%95%9C-%EA%B3%A0%EC%B0%B0/)
  
# Basic

Sticky Session 은 Load Balancer 가 Client 의 요청을 특정 Server 에게만 Routing
되도록 하는 기술이다. 

첫번재 요청에 대해 Server 는 자신의 정보를 Cookie 에 담아서 Client 에게 HTTP
응답한다. 이후 Client 의 HTTP Request Cookie 는 특정 Server 정보를 담고 있게
된다. Load Balancer 는 Client 의 HTTP Request Cookie 를 확인하고 특정 Server
에게만 Routing 한다.

Sticky Session 을 사용하면 Load Balancer 가 공평하게 Routing 한다고 볼 수 없다.

Server 가 session 관리를 각자 하기로 했다면 Sticky Session 이 도움이 될 수 있다. 그러나 모든 Server 가 Redis 에 session data 를 저장하는 것을 추천한다. Redis 는 Replication 을 지원하기 때문에 SPOF 가 아니다. 
