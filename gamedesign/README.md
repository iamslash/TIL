# Abstract

게임을 제작하기 위한 전반적인 기술들을 정리한다.

# Material

* [Torchlight2 GUTS](http://www.moddb.com/games/torchlight-ii/news/torchlight-ii-guts)
  * steam | library | tools 에서 다운로드 하자.
* [Torchlight2 GUTS document](http://docs.runicgames.com/wiki/Main_Page)  
  * 위키에 잘 정리되어 있다.
* [Torchlight2 GUTS tutorial @ youtube](https://www.youtube.com/watch?v=zkJBQb_64rM&list=PLuT4OTqZoR4mUDpjGC-s3toYZIPpNnhWR)

# Document

* [game design document](/gamedesigndoc/README.md)

# Level Editor

* 토치라이트2의 레벨에디터인 GUTS가 최고다.

## CRUD object

* 캐릭터
  * name, face, char status, char skill
* 캐릭터 상태
  * LV, HP, MAXHP, RARRITY, STAR
* 캐릭터 상태 성장패턴
  * Linear, Parabola
* 캐릭터 스킬 (캐릭터 어펙트를 생성)
  * melee, arrow, gun 
* 캐릭터 상태 이상 (캐릭터 어펙트를 생성)
  * name, duration, affect
* 프로젝타일 (캐릭터 어펙트를 생성)
  * energ ball, fire ball
* 캐릭터 어펙트
  * HP감소, 1초마다 HP감소
* 캐릭터 상태 이상 (캐릭터 어펙트를 생성)
  * HP증가
* 아이템 (consume, wear)
  * name, reqlv, type, item status
* 아이템 상태
  * LV, HP, MAXHP, RARITY, STAR
* 아이템 상태 성장패턴
  * Linear, Parabola
