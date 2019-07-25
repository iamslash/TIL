- [Abstract](#abstract)
- [Materials](#materials)
- [Terms](#terms)
  - [Storage](#storage)
  - [as a service](#as-a-service)
  - [...](#)
- [Basic](#basic)
  - [VPC (Virtual Private Cloud)](#vpc-virtual-private-cloud)
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

# Terms

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
  
## ...


waf
osi 7 layer
onpremise 
vpc
restfull api
msa
good architect
hdfs 
virtualization 3 가지
devops msa 관계
how to make a vpc on onpremise
CAP
PACELC
capa
CIDR
subnet

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

