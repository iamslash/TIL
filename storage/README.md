# Abstract

Computer Data Storage 에 대해 정리한다.

# Materials

# Overview

* [SAN의 정의 그리고 NAS와의 차이점](http://www.ciokorea.com/news/37369)

----

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
* RAID - Redundant Array of Inexpensive/Independent Disk
  * 저장 장치 여러 개를 묶어 고용량·고성능 저장 장치 한 개와 같은 효과를 얻기 위해 개발된 기법이다.
  * RAID 0 (Stripping)
    * 여러 개의 멤버 하드디스크를 병렬로 배치하여 거대한 하나의 디스크처럼 사용한다. 데이터 입출력이 각 멤버 디스크에 공평하게 분배되며, 디스크의 수가 N개라면 입출력 속도 및 저장 공간은 이론상 N배가 된다. 다만 멤버 디스크 중 하나만 손상 또는 분실되어도 전체 데이터가 파손되며, 오류검출 기능이 없어 멤버 디스크를 늘릴수록 안정성이 떨어지는 문제가 있다.
  * RAID 1 (Mirroring)
    * 각 멤버 디스크에 같은 데이터를 중복 기록한다. 멤버 디스크 중 하나만 살아남으면 데이터는 보존되며 복원도 1:1 복사로 매우 간단하기 때문에, 서버에서 끊김 없이 지속적으로 서비스를 제공하기 위해 사용한다. 
  