# Scalability

서비스의 사용량이 늘어나면 system, process, network 등등 서비스의 성능도 늘려야 하는 것을 말한다.

CPU, RAM, Storage 를 조정하는 것을 vertical scaling 이라한다. machine 의 개수를 조정하는 것을 horizontal scaling 이라고 한다.

# Reliability

시스템의 신뢰도를 의미한다. 곧, 장애가 발생하더라도 시스템에 큰 영향이 없어 서비스의 신뢰도를 말한다.

distributed systemd 에서 redundancy 를 이용하여  reliability 를 실현한다. 예를 들어 유저가 쇼핑몰에서 장바구니에 담아놓은 데이터를 중복해서 보관한다면 하나의 서버가 장애가 났을 때 다른 서버를 이용하여 분산 시스템의 신뢰성을 실현할 수 있다.

# Availability

시스템의 가능한 정도를 의미한다. 곧, 서비스를 문제없이 이용할 수 있는 기간을 말한다.

Reliability 와 비슷하지만 Reliability 는 장애가 발생했을 때에 대한 것이고 Availability 는 평소에 대한 것이다.
예를 들어 쇼핑몰의 서비스가 일년동안 단 한번의 장애가 없었지만 이년째에 장애가 발생하여 복구가 힘들었다면 availability 는 높지만 reliability 는 낮다고 할 수 있다.

# Efficiency

response time (latency), throughput (bandwidth) 으로 시스템의 효율성을 판단할 수 있다.

latency 는 시스템이 얼만큼 응답지연이 있는지를 말한다. bandwidth 는 시스템이 얼만큼 데이터를 이동시켰는지를 말한다.

# Serviceability or Manageability

얼마나 쉽고 빠르게 서비스를 유지보수할 수 있는지를 말한다.

* 장애를 빠르게 파악할 수 있어야 한다.
* 장애가 발생했을 때 빠르게 복구할 수 있어야 한다. 
* 업데이트를 쉽고 빠르게 할 수 있어야 한다.