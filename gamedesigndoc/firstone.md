- [Overview](#overview)
- [Story](#story)
- [Contents](#contents)
    - [Conents Map](#conents-map)
    - [Village](#village)
    - [Worldmap](#worldmap)
    - [Dungeon](#dungeon)
    - [Store](#store)
    - [Inventory](#inventory)
    - [Smith](#smith)
- [Controlls](#controlls)
- [Stats](#stats)
- [Characters](#characters)
    - [Character Stats](#character-stats)
- [Items](#items)
    - [Items Stats](#items-stats)
- [Skills](#skills)
    - [Skill Stats](#skill-stats)
- [Damage](#damage)
    - [Damage Formula](#damage-formula)
    - [Damage Per Seconds Formula](#damage-per-seconds-formula)
- [Affect](#affect)
- [Altered Status](#altered-status)
- [Assets](#assets)
    - [App icon](#app-icon)
    - [Promotion Resources](#promotion-resources)
    - [Datas](#datas)
    - [2D GUI](#2d-gui)
    - [2D GUI Effect Particles](#2d-gui-effect-particles)
    - [2D Characters](#2d-characters)
    - [2D Environments](#2d-environments)
    - [3D Character Models](#3d-character-models)
    - [3D Character Textures](#3d-character-textures)
    - [3D Character Animations](#3d-character-animations)
    - [3D Environment Models](#3d-environment-models)
    - [3D Environment Textures](#3d-environment-textures)
    - [3D Environment Animations](#3d-environment-animations)
    - [3D Effect Particles](#3d-effect-particles)
    - [3D Projectiles](#3d-projectiles)
    - [Sound](#sound)
    - [Video](#video)
- [Schedules](#schedules)

---

# Overview

숲속에서 핵앤슬래시 하는 게임. 기획서는 소설책 처럼 읽힐 수 있도록 작성해주면 좋을 것 같다. 엑셀에 이것 저것 두서 없이 도표가 난무하는 형태는 가독성이 매우 떨어진다.

# Story

어쩌구 저쩌구 블라 블라 블라...

# Contents

## Conents Map

화면과 연결선들의 모음. 간략한 설명을 통해 컨텐츠를 이해한다. 

## Village

## Worldmap

## Dungeon

## Store

## Inventory

## Smith 

# Controlls

전투를 조작하는 방식

# Stats

character, item, skll 등이 stat을 가질 수 있다. 보통 static stat, dynamic stat등으로 분류한다. 

static stat은 basic stat 등으로 분류한다. 처음 태어날 때 한번 정해지면 변하지 않는다.

dynamic stat은 growth stat, combat stat 등으로 분류한다. growth stat은 시스템에 의해 변화한다. combat stat 은 전투중일 때 상태이상에 의해 수시로 변한다. 

# Characters

**[^1]캐릭터의 스탯은 성장 스탯과 전투 스탯으로 나누어 진다.** 성장 스탯은
얼만큼 성장 했는지를 표현해준다. 예를 들어 level, grade, rarity, star
등등이 있다. 전투 스탯은 캐릭터가 전투를 수행할 때 사용되는
스탯이다. 성장 스탯과 함께 계산되어 최종 결과를 전투에서 사용한다.

**기획자의 의도는 항상 볼드체로 표시하자.** 미리 보기가 있으니 너무
편한 걸 어떡하지? 뭔가 단락을 나눠주는 그런 도구가 있다면 더욱 좋을 것
같다.

캐릭터는 무기, 방어구, 장신구를 소유할 수 있다. 무기, 방어구는 한개씩 장신구는 3개를 소유할 수 있다.

[^1]: 이것은 첨자의 예이다.

## Character Stats

basic stat, growth stat, combat stat 등으로 나눌 수 있다. basic stat은 캐릭터의 탄생과 관계가 있다. growth stat은 캐릭터의 성장과 관계가 있다. combat stat은 캐릭터의 전투와 관계가 있다. 

basic stat은 한번 정해지면 변하지 않는다. 그러나 growth stat, combat stat, 등은 상황에 따라 변한다.

다음은 basic stat의 종류이다.

| stat | desc |
|:----:|:----:|
| STR | ... |
| DEX | ... |
| ITL | ... |
| FOC | ... |

다음은 growth stat의 종류이다.

| stat | desc |
|:----:|:----:|
| LV | ... |
| RARITY | ... |
| GRADE | ... |
| STAR | ... |

다음은 combat stat의 종류이다.

| stat | desc | formula |
|:----:|:----:|:---:|
| HP | ... | |
| MP | ... | |
| ATK | ... | LV * 100 |
| MAXHP | ... | LV * 100 |
| MAXMP | ... | LV * 100 |
| CRIT | ... | LV / 100 |
| DODGE | ... | |
| SPD | ... | |
| bbb | ... | |
| ccc | ... | |

# Items

wearable item, consumable item, pluggable item 등으로 분류한다. wearable item 은 다시 weapon, armor, accesorry 등으로 분류한다. pluggable item 은 crytal, rune 등이 있다. 

## Items Stats

# Skills

active skill, passive skill 등으로 분류한다.

active skill은 다음과 같다.

| stat | desc |
|:----:|:----:|
| melee | ... |
| range | ... |

passive skill은 다음과 같다.

| stat | desc |
|:----:|:----:|
| bless | ... |
| mighty |  |

## Skill Stats

# Damage

## Damage Formula

`Damage = LV * CRIT * 100`

## Damage Per Seconds Formula

`DPS = Damage * SPD`

# Affect

skill, item등에 의해 발생된다.

# Altered Status

상태이상의 종류는 다음과 같다.

| stat | desc |
|:----:|:----:|
| 뇌격 | ... |
| 독 | ... |

# Assets

## App icon

## Promotion Resources

2D image, video etc,...

## Datas

game, lang, slang, etc...

## 2D GUI

## 2D GUI Effect Particles

## 2D Characters

## 2D Environments

## 3D Character Models

## 3D Character Textures

## 3D Character Animations

## 3D Environment Models

## 3D Environment Textures

## 3D Environment Animations

## 3D Effect Particles

## 3D Projectiles

## Sound

## Video

# Schedules

언제까지 끝낼 것인가... gantt chart가 제일 좋다.