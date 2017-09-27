# Abstract

- 3d그래픽의 라이팅에 대해 기술한다.

# Contents

* [Terms](#terms)
* [PBR](#pbr)

# Learning Materials

- [Real Shading in Unreal Engine 4](https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf)
  - 2013 SIGRAPH paper
  - [번역](http://lifeisforu.tistory.com/348)
- [A Reflectance Model for Computer Graphics](http://graphics.pixar.com/library/ReflectanceModel/paper.pdf)  
  - [번역](http://lifeisforu.tistory.com/349)
- [Physically-Based Shading at Disney](https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf)  
  - [번역](http://lifeisforu.tistory.com/350)
- [Microfacet Models for Refraction through Rough Surfaces](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf)  
  - [번역](http://lifeisforu.tistory.com/352)
- [PBR 이란 무엇인가 @ tistory](http://lifeisforu.tistory.com/366)

# Fundamentals

## light

![전자기파의 스펙트럼](img/EM_spectrum.png)

빛은 전자기파(Electromagnetic wave)이다. 전자기파는 전자기적 과정에 의해 발생하는
에너지이다. 파동의 형태로 발산되는 에너지라고 할 수 있다. 빛은 파장이 다른 여러 가지 
전자기파로 구성되어 있다.

![](img/Light_dispersion_conceptual_waves350px)

빛은 여러가지 전자기파들로 구성되어 있기 때문에 프리즘을 사용해서 각각을 분리할 수 있다.
빨간색 파장의 길이가 가장 길고 보라색 파장의 길이가 가장 짧다.

## Eye

![인간의 눈 구조](img/HumanEye.png)

우리 눈의 망막(retina)를 확대해 보면 원추세포(cone cell: blue cone,
red cone, green cone)와 간상세포(rod)가 있다. 원추세포(cone cell)는
색상을 감지하고 간상세포(rod)는 명암을 감지한다.  원추세포는 그것을
구성하고 있는 감광세포의 종류에 따라 받아들일 수 있는 색상이 정해져
있다. 우측 그림을 보면 3종류의 원추세포가 존재하고 그것은 RGB와 같다.
이것은 텍스처를 제작할때 RGB채널을 사용하는 이유기도 하다.

원추세포(cone cell)는 약 600만개이고 간상세포(rod)는 약 9000만개이다. 
따라서 우리 눈은 색상 대비보다 명도 대비에 더욱 민감하다.

## 조도와 휘도 (illuminance & luminance)

## 빛의 감쇠 (attenuation)

## 광원의 밝기 - 광속 (luminous flux)

## 조도 측정

## 휘도 측정

# Lambertian Reflectance Model

# Half Lambert Diffuse

# Phong Reflectance  Model

# Rim Lighting

# Cook-Torrance Model

# Oren-Nayar Model

# [PBR](../pbr/README.md)

# Ray Traciing

# Radiosity

