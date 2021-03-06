# Abstract

- GIT을 쉽게 사용할 수 있도록 문서를 작성한다.

# Contents

* [Concept](#concept)
* [Contingency](#contingency)
* [Public Key, Private Key](#public-key-private-key)
* [Prerequisites](#prerequisites)

# Concept

- 게임의 세이브 포인트 처럼 내가 작업 하는 내역들이 저장된다면 얼마나
  좋을까? 현재 작업을 열심히 하면서 만들어진 파일들을 뒤로 하고 어제
  저장한 세이브 포인트로 한번에 변신할 있다면 얼마나 좋을까?  그런
  고민을 위해서 탄생한 프로그램이 git이다. 물론 git말고도 무수히 많은
  버전 관리 프로그램들이 넘쳐난다. 하지만 대세는 git이다.
- git을 개발자들이 공부하는 방식으로 접근하면 다소 러닝 커브가 높다고
  할 수 있다. 하지만 sourtree와 같은 응용 프로그램을 활용한다면
  어렵기만 했던 git을 잘 활용할 수 있다.
  
# Contingency

- 대체 git은 무엇이란 말인가? 간단히 상황극을 설정해서 어떻게 git이
  동작하는지 가볍게 살펴보자. 
- A라는 사람과 B라는 사람이 있다. 그리고 프로젝트 L이 담겨있는
  디렉토리가 있다. 프로젝트 L이 담겨있는 디렉토리는 저 멀리 서버에
  저장되어 있다. 그래야 A와 B는 똑같은 내용을 다운로드 받아서 작업을
  할 것이다.
- A, B는 서버에 담겨 있는 프로젝트 L을 다운로드 받아서 c:\prj\L에
  저장했다.  A는 한국에 있고 B는 미국에 있다. 둘은 메신저로 협업하는
  중이다. 
- A는 매니저로부터 작업지시를 받아 하나의 일을 완성했다. 그것은
  1.png에 모두 담겨있다. 그리고 두번째 작업지시를 받았다. 1.png에
  수정사항을 반영했다.  그런데 매니저로부터 다시 요청이
  날아왔다. 두번째 작업지시는 하지 말았으면 좋겠으니 그전의 1.png에서
  새로운 작업을 해주었으면 좋겠다는 요청 사항이다. A는 난감하다.  예전
  상황으로 돌아갈 방법이 있었으면 좋겠다. git은 이런 요구사항을 정확히
  만족시켜준다.
- A가 먼저 a.png를 저장했다. 그리고 서버에 업로드 했다. B는 A에게 연락
  받고 a.png를 다운로드 했다. 그리고 몇가지 수정을 한 후 서버에 업로드
  했다. 그 사이 A도 a.png를 수정했다. 그리고 몇가지 수정을 한 후
  서버에 업로드 했다. 짜잔 과연 이후에 무슨일이 발생 했을까? A가
  수정한 a.png와 B가 수정한 a.png는 서버에 옳바르게 저장되어 있을까?
  그렇지 않다. B가 마지막에 수정한 a.png가 덮어쓰기 했으므로 A가
  두번째 작업한 내역은 서버의 a.png에 저장되어 있지 않다.  이것을
  충돌했다라고 표현한다. git은 이런 충돌 현상을 해결할 수 잇게 해준다.
- 앞서 언급한 상황극에서 발생한 난감한 상황들은 git에 의해 모두 해결
  가능하다. 어떻게 그럴 수 있을까? 먼저 git 세계관에 등장하는 몇가지 
  주춧돌들을 알아야 한다. 그 주춧돌은 다음과 같다.
  - working directory (작업 저장소)
  - STAGE (대기소)
  - local repository (가까운 저장소)
  - server repository (머나먼 저장소)
- 앞서 언급한 상황극에서 등장했던 프로젝트 L이 저장되어 있는 서버의
  디렉토리를 우리는 server repository라고 한다. 이제부터 우리는 머나먼
  저장소라고 이름붙여 보자. A가 작업한 c:\prj\L은 working
  directory라고 한다. 이제 부터 작업 저장소 라고 부르자. A가 머나먼 저장소로부터
  git을 이용해서 어떻게 어떻게 다운로드 받아서 c:\prj\L에 저장하면 
  c:\prj\L\.git라는 디렉토리가 생성된다. 이것을 local repository라고 한다.
  이제부터 가까운 저장소라고 부르자.
- A가 무언가 작업을 하면 작업 내역은 당연히 작업 저장소에 남겨질
  것이다. 예를 들어 a.png를 만들면 자업 저장소에 a.png가 남게 된다. 이것을
  가까운 저장소에 일단 저장해야 한다. 이것은 마치 게임에서 세이브 포인트에
  내가 작업한 내역을 저장한 것과 같다. 한 발 더 나아가서 B에게 내가 작업한
  내역을 알리려면 머나먼 저장소에 가까운 저장소에 저장했던 것을 업로드 해야 한다.
  이것이 git이 버전관리하는 대강의 흐름이다.

# public key, private key

- 머나먼 저장소는 누구나에게 프로젝트 L에 대해 접근권한을 줘야 할까?
  예를 들어서 A, B와는 전혀 관련 없는 C라는 친구가 있다고 하자. 
  C는 A의 고교 동창이다. A와 술을 먹다가 프로젝트 L에 대해 이야기를 
  들었다. C는 나도 한번 프로젝트 L을 보고 싶다고 생각하고 머나먼
  저장소에서 다운로드 시도를 해보았다. 하지만 절대 그런일은 가능하지 않았다.
  바로 공개키, 개인키 때문이다. 
- A는 프로젝트 L을 머나먼 저장소로 부터 다운로드 받기 위해 키를
  만들었다.  sourcetree를 이용해서 public key, private key 이렇게
  두가지를 만들었다. 이제부터 공개키, 개인키라고 하자. 둘다 파일의 형태이다.
  개인키는 절대 누구에게도 유출되면 안되는 파일이다. 공개키는 누구에게
  유출되도 상관없다. 마나먼 저장소가 포함되어 있는 서버에 A의 공개키를
  등록해야 한다. 
- 그리고 A는 자신의 개인키를 머나먼 저장소가 포함되어 있는 서버에 보여주고
  서버는 A의 개인키와 짝을 이루는 공개키가 등록되어 있는지 검사후에
  자격여부를 판단하여 다운로드를 허락한다. sourcetree는 puttegen.exe라는
  다른 프로그램을 실행해서 유저가 공개키, 개인키를 제작 할 수 있도록 해준다.
- (실습)개인키 공개키를 만들어 보자.
  
# Prerequisites

- 먼저 git을 사용하려면 몇가지 준비가 필요하다. 머나먼 저장소를 제작하기 위해
  우리는 gitlab이라는 서비스를 이용한다. 이곳에서 프로젝트 L을 만들고
  나의 공개키를 등록하자.
- (실습)gitlab가입, 프로젝트제작, 공개키 등록
- 그리고 git에 GUI를 입힌 sourcetree를 설치하자.
- (실습)sourcetree설치

# CLONE (머나먼 저장소에서 복사하기)

- 프로젝트 L을 시작하려면 머나먼 저장소로부터 프로젝트를 다운로드
  받아야 한다. 최초 한번만 하면 되는 절차이다. 이것을 CLONE이라고 한다.
- (실습)clone하기

# ADD (변경사항 대기시키기)

- c:\prj\L에 a.png를 추가합시다. 자 이것을 게임의 세이브 포인트 처럼
  저장하고 싶다. 내가 변경한 사항들을 STAGE에 올려봅시다.
- (실습)ADD하기 STAGE확인 하기  
 
# COMMIT (가까운 저장소에 저장하기)

- 게임의 경우 내가 저장한 것은 슬롯의 형태로 표현됩니다.  git의 경우는
  무수히 많은 슬롯이 존재합니다. 이 슬롯을 REVISION이라고 합니다.
  세계에서 유일 무이한 고유의 아이디가 하나 발급됩니다. 이 아이디만
  알면 내가 저장한 슬롯으로 언제든지 돌아갈 수 있습니다. 아이디가 좀
  깁니다요.
- COMMIT을 하면 STAGE의 변경 이력들이 REVISION을 하나 만들면서 가까운
  저장소에 저장됩니다. REVISION그래프에 REVISION이 하나 만들어지는
  순간이다.
- (실습)COMMIT하기 REVISION확인하기

# PUSH (머나먼 저장소에 저장하기)

- 가까운 저장소에 저장된 내용을 머나먼 저장소로 업로드 하는 것을 
  PUSH라고 합니다.
- (실습)PUSH하기 

# PULL (FETCH, MERGE)

- B는 이제 A가 작업한 내역을 머나먼 저장소로 부터 다운로드 받아서
  가까운 저장소에 저장하고 작업 저장소에 적용하고 싶습니다. 이것을
  PULL이라고 합니다.
- (실습)PULL하기  

# FETCH (머나먼 저장소에서 내려받아 가까운 저장소에 저장하기)

- 사실 PULL은 FETCH라는 것과 MERGE라는 것이 함께 수행되는 것입니다.
  머나먼 저장소에서 가까운 저장소로 다운로드하는 것을 FETCH라고 합니다.
- (실습)FETCH하기  

# CHECKOUT (변신하기)

- COMMIT을 두번 하면 두개의 REVISION이 만들어집니다.
  첫번째 REVISION으로 돌아가 봅시다. 이것을 checkout이라고 합니다.
- (실습)checkout하기 

# CONFLICT (충돌)

- A가 a.png를 수정하고 B도 a.png를 수정했습니다.  A가 먼저 ADD,
  COMMIT, PUSH후예 b가 Add, COMMIT, push하면 머나먼 저장소로 부터
  업로드가 거부당합니다. 이것은 충돌이 발생했기 때문입니다.
  간단히 충돌을 해결해 봅시다.
- A의 가 작업한 파일 a.png에 내가 작업한 내역을 다시 적용하여
    COMMIT, PUSH합니다.
- (실습)충돌해결하기

# BRANCH (작업 흐름)

- A는 자신이 작업한 것과 다른 사람이 작업한 것이 충돌 되지 않았으면
  좋겠습니다. 그래서 Branch를 제작합니다. Branch의 이름은
  "feature/A/make_nice_b.png" 라고 지었습니다.
- Branch는 revision그래프의 다른 줄기입니다.  
- (실습)브랜치 만들기 

# MERGE (가까운 저장소에서 작업 장소에 합치기)

- A, B의 매니저는 "feature/A/make_nice_b.png"와
  "feature/B/make_nice_c.png"를 master브랜치에 합쳐서
  적용합니다. 이것을 MERGE라고 합니다.
- (실습)master에 merge하기
