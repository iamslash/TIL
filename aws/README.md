- [Abstract](#abstract)
- [Materials](#materials)
- [Prerequisites](#prerequisites)
  - [Storage](#storage)
  - [as a service](#as-a-service)
  - [WAF (Web Application Fairewall)](#waf-web-application-fairewall)
  - [XSS (Cross Site Scripting)](#xss-cross-site-scripting)
  - [CSRF (Cross Site Request Forgery)](#csrf-cross-site-request-forgery)
  - [XSS vs CSRF](#xss-vs-csrf)
  - [OSI 7 layer](#osi-7-layer)
  - [TCP/IP Protocol Suite](#tcpip-protocol-suite)
  - [Subnet](#subnet)
  - [CIDR (Classless Inter-Domain Routing)](#cidr-classless-inter-domain-routing)
  - [Private Network](#private-network)
  - [VPN (Virtual Private Network)](#vpn-virtual-private-network)
  - [VPC (Virtual Private Cloud)](#vpc-virtual-private-cloud)
  - [RESTfull API](#restfull-api)
  - [MSA](#msa)
  - [HDFS](#hdfs)
  - [Virtualization](#virtualization)
  - [Devops](#devops)
  - [CAP](#cap)
  - [PACELC](#pacelc)
- [Basic](#basic)
  - [VPC (Virtual Private Cloud)](#vpc-virtual-private-cloud-1)
  - [EC2 (Elastic Compute)](#ec2-elastic-compute)
  - [IAM (Identity and Access Management)](#iam-identity-and-access-management)
  - [CloudFront](#cloudfront)
  - [S3 (Simple Storage Service)](#s3-simple-storage-service)
  - [RDS](#rds)
  - [ElastiCachi](#elasticachi)
  - [Lambda](#lambda)
  - [API Gateway](#api-gateway)
  - [DynamoDB](#dynamodb)
  - [ElasticSearch Service](#elasticsearch-service)
  - [Kinesis](#kinesis)
  - [Route 53](#route-53)
  - [CloudWatch](#cloudwatch)
  - [ELB](#elb)
- [Advanced](#advanced)
  - [How to use awscli on Windows](#how-to-use-awscli-on-windows)
- [Best Practices](#best-practices)

----

# Abstract

aws 사용법에 대해 간략히 정리한다.

# Materials

* [AWS Services Overview - September 2016 Webinar Series @ slideshare](https://www.slideshare.net/AmazonWebServices/aws-services-overview-september-2016-webinar-series)
* [AWS @ 생활코딩](https://opentutorials.org/course/2717)
* [44bit](https://www.44bits.io/ko)
  * 클라우드블로그
* [아마존 웹 서비스를 다루는 기술](http://pyrasis.com/aws.html)
  * 오래되었지만 괜찮은 책
* [cloudcraft](https://cloudcraft.co/)
  * aws diagram tool
* [Amazon Web Services 한국 블로그](https://aws.amazon.com/ko/blogs/korea/tag/korea-techtips/)
* [AWS re:Invent 2018 DeepDive series @ youtube](https://www.youtube.com/results?search_query=AWS+re%3AInvent+2018+Deep+Dive)
* [AWS re:Invent 2018 Under the hood series @ youtube](https://www.youtube.com/results?search_query=AWS+re%3AInvent+2018+Under+the+hood)
* [AWS Summit 2019 | AWS 아키텍처 @ youtube](https://www.youtube.com/playlist?list=PLORxAVAC5fUWWpxC5TW10P35GpLBOKWKS)
* [AWS Summit 2019 | 게임 @ youtube](https://www.youtube.com/playlist?list=PLORxAVAC5fUXgqiJJKJ8h8E5BkrFXUS2j)
* [AWS Summit 2019 | AWS 기술트랙 1 @ youtube](https://www.youtube.com/watch?v=KKkK6d8Srik&list=PLORxAVAC5fUUoQB13KiV8ezs7cAfwSagC)
* [AWS Summit 2019 | AWS 기술트랙 2 @ youtube](https://www.youtube.com/watch?v=l7W_urK43aE&list=PLORxAVAC5fUX3c9KwLE9E-qZv7tXk-b3O)
* [AWS Summit 2019 | AWS 기술트랙 3 @ youtube](https://www.youtube.com/watch?v=QFeSXY3cL7Q&list=PLORxAVAC5fUUa8XFFLtB6aK4vgZhTKBLg)
* [AWS Summit 2019 | AWS 기술트랙 4 @ youtube](https://www.youtube.com/watch?v=Sf6j7PPHeeI&list=PLORxAVAC5fUUeaSHb91d5wpDGfR14uNCi)
* [AWS Summit 2019 | AWS 기술트랙 5 @ youtube](https://www.youtube.com/watch?v=nxgGk-PbXf0&list=PLORxAVAC5fUWZGawyaMyz8NepNGqHbHtZ)
* [AWS Summit 2019 @ youtube](https://www.youtube.com/playlist?list=PLORxAVAC5fUWyB6Hsk9ibYJHw97k1h6s9)

# Prerequisites

## Storage 

* [SAN의 정의 그리고 NAS와의 차이점](http://www.ciokorea.com/news/37369)

* NAS - Network Attatched Storage
  * 표준 이더넷 연결을 통해 네트워크에 부착된 저장장치.
* DAS - Direct Attatched Soorage
  * 머신에 부착된 저장장치
* SAN - Storage Area Network
  * 파이버 연결 채널을 통해 네트워크에 고속으로 부착된 저장장치.
* SAN vs NAS
  * SAN 과 NAS 는 모두 네트워크 기반 스토리지 이다. 그러나 SAN 은 일반적으로 파이버 채널 연결을 이용하고 NAS 는 표준 이더넷 연결을 통해 네트워크에 연결된다. 
  * SAN은 블록 수준에서 데이터를 저장하지만 NAS는 파일 단위로 데이터에 접속한다. 
  * 클라이언트 OS 입장에서 보면, SAN 은 일반적으로 디스크로 나타나며 별도로 구성된 스토리지용 네트워크로 존재한다. 반면 NAS 는 클라이언트 OS 에 파일 서버로 표시된다.
* Unified storage
  * SAN 과 NAS 가 합쳐진 것이다.
  * iSCI (Internet Small Computing System Interface), NFS, SMB 모두를 지원하는 Multiprotocol Storage 이다.

## as a service

* [SaaS vs PaaS vs IaaS: What’s The Difference and How To Choose](https://www.bmc.com/blogs/saas-vs-paas-vs-iaas-whats-the-difference-and-how-to-choose/)

![](img/saas-vs-paas-vs-iaas-810x754.png)

* On-Premises
  * Netwoking 부터 Appliations 까지 유저가 모두 관리해야 하는 개발 환경
* IaaS (Infrastructure as a service)
  * Networking 부터 Virtualization 까지 유저대신 관리해주는 서비스 이다. 유저는 O/S 부터 Applications 까지 관리한다.
  * DigitalOcean, Linode, Rackspace, Amazon Web Services (AWS), Cisco Metapod, Microsoft Azure, Google Compute Engine (GCE)
* PaaS (Platform as a service)
  * Networking 부터 Runtime 까지 유저대신 관리해주는 서비스 이다. 유저는 Data 부터 Applications 까지 관리한다.
  * AWS Elastic Beanstalk, Windows Azure, Heroku, Force.com, Google App Engine, Apache Stratos, OpenShift
* SaaS (Software as a service)
  * Networking 부터 Applications 까지 유저대신 관리해주는 서비스 이다. 유저는 별도로 관리할 필요가 없다.
  * Google Apps, Dropbox, Salesforce, Cisco WebEx, Concur, GoToMeeting
  
## WAF (Web Application Fairewall)

* [AWS WAF – 웹 애플리케이션 방화벽](https://aws.amazon.com/ko/waf/)
* [웹방화벽이란?](https://www.pentasecurity.co.kr/resource/%EC%9B%B9%EB%B3%B4%EC%95%88/%EC%9B%B9%EB%B0%A9%ED%99%94%EB%B2%BD%EC%9D%B4%EB%9E%80/)
  
* 일반적인 방화벽과 달리 웹 애플리케이션의 보안에 특화된 솔루션이다. 
* 애플리케이션의 가용성에 영향을 주거나, SQL Injection, XSS (Cross Site Scripting) 과 같이 보안을 위협하거나, 리소스를 과도하게 사용하는 웹 공격으로부터 웹 애플리케이션을 보호하는 데 도움이 된다.

## XSS (Cross Site Scripting)

* [웹 해킹 강좌 ⑦ - XSS(Cross Site Scripting) 공격의 개요와 실습 (Web Hacking Tutorial #07) @ youtube](https://www.youtube.com/watch?v=DoN7bkdQBXU)
* 웹 게시판에 javascript 를 내용으로 삽입해 놓으면 그 게시물을 사용자가 읽을 때 삽입된 스크립트가 실행되는 공격방법

## CSRF (Cross Site Request Forgery)

* [웹 해킹 강좌 ⑩ - CSRF(Cross Site Request Forgery) 공격 기법 (Web Hacking Tutorial #10) @ youtube](https://www.youtube.com/watch?v=nzoUgKPwn_A)
* 특정 사용자의 세션을 탈취하는 데에는 실패하였지만 스크립팅 공격이 통할 때 사용할 수 있는 해킹 기법. 피해자가 스크립트를 보는 것과 동시에 자기도 모르게 특정한 사이트에 어떠한 요청(Request) 데이터를 보낸다.

## XSS vs CSRF

* XSS 는 공격대상이 Client 이고 CRSF 는 공격대상이 Server 이다.
* XSS 는 사이트변조나 백도어를 통해 Client 를 공격한다.
* CSRF 는 요청을 위조하여 사용자의 권한을 이용해 서버를 공격한다.

## OSI 7 layer

![](/network/Osi-model-7-layers.png))

ISO (International Standard Organization) 에서 개발한 네트워크 모델이다. 컴퓨터 네트워크 프로토콜 디자인과 통신을 계층으로 나누어 설명했다. 1984 년에 ISO 가 ISO 7498, CCITT (ITU-T) 는 X.200 으로 출판하였다.

## TCP/IP Protocol Suite

* [Computer Network | TCP/IP Model @ geeksforgeeks](https://www.geeksforgeeks.org/computer-network-tcpip-model/)

![](img/tcpAndOSI.png)

1960 년대 말 DARPA (Defense Advanced Research Projects Agency) 가 수행한 연구개발의 산출물로 부터 탄생했다. DARPA 가 개발한 ARPANET 는 NCP (Network Control Program) 라는 프로토콜을 사용했다. 1983 년 TCP/IP 가 이를 대체하며 지금의 인터넷으로 진화했다.

Internet Protocol Suite 는 인터넷에서 컴퓨터들이 서로 정보를 주고 받는데 쓰이는 프로토콜의 모음이다. TCP, IP 가 가장 많이 쓰이기 때문에 TCP/IP Protocol Suite 라고도 한다. 

TCP/IP Protocol Suite 는 OSI 의 7 개 레이어 를 4 개로 줄였다.

## Subnet

IP 의 network 가 논리적으로 분할된 것이다. Subnet 을 만드는 것을 Subnetting 이라고 한다. Subnet 은 Subnet Mask 를 이용하여 만든다. 

Subnet 의 IP address 는 network number 혹은 routing prefix 와 rest field 혹은 host identifier 와 같이 두가지로 나뉘어 진다. 

routing prefix 는 `198.51.100.0/24` 와 같이 CIDR (Classless Inter-Domain Routing) 표기법으로 표현할 수 있다. `198.51.100.0/24` 의 subnet mask 는 `255.255.255.0` 이다.

## CIDR (Classless Inter-Domain Routing)

기존의 IP 주소 할당 방식이었던 네트워크 클래스를 대체하기 위해 IETF ()가 1993년 에 [RFC4632](https://tools.ietf.org/html/rfc4632) 라는 표준으로 개발한 IP 주소 할당 방법이다. 

`198.51.100.0/24` 와 같이 표기한다. `/24` 는 routing prefix 비트의 개수를 의미한다.

## Private Network

Private IP Address 를 사용하는 네트워크이다. IETF (Internet Engineering Task Force) 는 Private IP Address 를 위해 다음과 같은 IPv4 주소를 사용하도록 RFC 1918 라는 표준으로 IANA ( Internet Assigned Numbers Authority) 를 지도했다.

| RFC 1918 name | IP address range | Numger of addresses | Largest CIDR block (subnet mask) | Host ID size | Mask bits | Classful description | 
|--|--|--|--|--|--|--|
| 24-bit block | 10.0.0.0 - 10.255.255.255 | 16,777,216 | 10.0.0.0/8 (255.0.0.0) | 24 bits | 8 bits | single class A network |
| 20-bit block | 172.16.0.0 – 172.31.255.255 | 1,048,576 | 172.16.0.0/12 (255.240.0.0) | 20 bits | 12 bits | 16 contiguous class B networks |
| 16-bit block | 192.168.0.0 – 192.168.255.255 | 65,536 | 192.168.0.0/16 (255.255.0.0) | 16 bits | 16 bits | 256 contiguous class C networks |

* 24-bit block 은 `10.x.x.x, 24 / 8 bits` 으로 기억하자.
* 20-bit block 은 `172.x.x.x, 20 / 12 bits` 으로 기억하자.
* 16-bit block 은 `192.168.x.x, 16 / 16 bits` 으로 기억하자.

## VPN (Virtual Private Network)

* [가상 사설망](https://namu.wiki/w/%EA%B0%80%EC%83%81%20%EC%82%AC%EC%84%A4%EB%A7%9D)
* 물리 적으로 여러 곳에 퍼져있는 컴퓨터들을 인터넷 네트워크와 암호화 기술을 이용하여 하나의 네트워크로 구성하는 것

## VPC (Virtual Private Cloud)

* [Amazon VPC란 무엇인가? @ aws](https://docs.aws.amazon.com/ko_kr/vpc/latest/userguide/what-is-amazon-vpc.html)

* AWS 외부와는 격리된 가상의 사설 클라우드이다.
* IP 주소 범위와 VPC 범위를 설정하고 서브넷을 추가하고 보안 그룹을 연결한 다음 라우팅 테이블을 구성한다.

## RESTfull API

restfull api

## MSA
msa

## HDFS

## Virtualization

virtualization 3 가지

## Devops

devops msa 관계

## CAP

## PACELC


# Basic

## VPC (Virtual Private Cloud)

* [만들면서 배우는 아마존 버추얼 프라이빗 클라우드(Amazon VPC) @ 44BITS](https://www.44bits.io/ko/post/understanding_aws_vpc)

AWS 외부와는 격리된 가상의 사설 클라우드이다. EC2 를 실행하려면 반드시 VPC 가 하나 필요하다. VPC 를 생성하기 위해서는 반드시 다음과 같은 것들을 함께 생성해야 사용이 가능하다.

```
1 VPC
n 서브넷 Subnet
1 라우트 테이블 Route Table
1 네트워크 ACLNetwork ACL
1 시큐리티 그룹 Security Group
1 인터넷 게이트웨이 Internet Gateway
1 DHCP 옵션셋 DHCP options set
```

## EC2 (Elastic Compute)

OS 가 설치된 machine 이다.

## IAM (Identity and Access Management)

[참고](https://www.44bits.io/ko/post/first_actions_for_setting_secure_account)

사용자와 권한을 담당한다. 주요 항목은 다음과 같다.

* 루트 사용자의 액세스 키 삭제(Delete your root access keys)
* 루트 사용자의 MFA 활성화(Activate MFA on your root account)
* 개별 IAM 사용자 생성(Create individual IAM users)
* 그룹을 사용하여 권한 할당(Use groups to assign permissions)
* IAM 비밀번호 정책 적용(Apply an IAM password policy)

## CloudFront

* [AWS2 - CloudFront @ 생활코딩](https://www.youtube.com/playlist?list=PLuHgQVnccGMDMQ1my6bVT-BPoo0LvnQMa)

CloudFront 는 CDN (Contents Delivery Network) 이다. 예를 들어, 특정 region 의 S3 bucket 을 전세계 유저들이 지연없이 다운 받을 수 있도록 캐싱하는 역할을 한다. 캐싱역할을 하는 엣지서버는 이곳 저곳 설치되어 있다.

## S3 (Simple Storage Service)

* [실전 Amazon S3와 CloudFront로 정적 파일 배포하기 @ aws](https://aws.amazon.com/ko/blogs/korea/amazon-s3-amazon-cloudfront-a-match-made-in-the-cloud/)

주요 command line 은 다음과 같다.

```
aws s3 ls
aws s3 rm
aws s3 mb # make bucket
aws s3 rb # remove bucket
aws s3 cp
aws s3 mv
aws s3 sync
```

## RDS

mySQL 등을 사용할 수 있다.

## ElastiCachi

redis, memcached 를 사용할 수 있다.

## Lambda

* [[AWS]Lambda의 시작 - 'Hello World' 출력하기](https://gun0912.tistory.com/60)
* [AWS Lambda: 가볍게 시작하기](https://hyunseob.github.io/2017/05/27/aws-lambda-easy-start/)

서버설정은 필요 없이 비지니스 로직을 실행할 수 있다. 함수만 작성하면 끝이다. coldstart 문제는 없을까? 
API Gateway 에서 routing 설정을 해야 외부에서 HTTP request 할 수 있다.

## API Gateway

* [[AWS]API Gateway - RESTful API만들기](https://gun0912.tistory.com/63)

HTTP URL 을 routing 할 수 있다. 예를 들어 `HTTP GET /user` 를 수신하면 `Lambda Hello` 혹은 `Lambda World` 등으로 routing 할 수 있다.

## DynamoDB

key value DB 이다.

## ElasticSearch Service

ElasticSearch, logstash, kibana 를 이용할 수 있다.

## Kinesis

kafka 를 이용할 수 있다.

## Route 53

DNS server 이다.

## CloudWatch

## ELB

# Advanced

## How to use awscli on Windows

* [참고](https://www.44bits.io/ko/post/aws_command_line_interface_basic)

다음과 같이 설치한다.

```
choco install awscli
```

* [키발급 참고](https://www.44bits.io/ko/post/publishing_and_managing_aws_user_access_key)

액세스키, 시크릿키를 발급받는다. 다음과 같이 설정하고 접속한다.

```bash
$ aws configure
AWS Access Key ID [None]: AKIAJEXHUYCTEHM2D3S2A
AWS Secret Access Key [None]: 3BqwEFsOBd3vx11+TOHhI9LVi2
Default region name [None]:ap-northeast-2
Default output format [None]:
```

configure 된 내용은 `~/.aws/config, ~/.aws/credentials` 로 저장된다.

* `~/.aws/config`

```
[default]
region = ap-northeast-2
```

* `~/.aws/credentials`

```
[default]
aws_secret_access_key = AKIAJEXHUYCTEHM2D3S2A
aws_access_key_id = 3BqwEFsOBd3vx11+TOHhI9LVi2
```

# Best Practices

