<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Materials](#materials)
- [Workflow](#workflow)
- [Key Elements](#key-elements)
- [Light Rays](#light-rays)
- [Absorption and Scattering - Transparency and Translucency](#absorption-and-scattering---transparency-and-translucency)
- [Diffuse and Specular Reflection - Microfacet Theory](#diffuse-and-specular-reflection---microfacet-theory)
- [Color](#color)
- [BRDF](#brdf)
- [Energy Conservation (에너지 보존법칙)](#energy-conservation-에너지-보존법칙)
- [Fresnel Effect - F0 (Fresnel Reflectance at 0 Degrees)](#fresnel-effect---f0-fresnel-reflectance-at-0-degrees)
- [Conductors and Insulators - Metals and Non Metal](#conductors-and-insulators---metals-and-non-metal)
- [Linear Space Rendering](#linear-space-rendering)
- [Implementation](#implementation)
    - [Unity3d](#unity3d)
    - [Unrealengine](#unrealengine)

<!-- markdown-toc end -->

-------------------------------------------------------------------------------

# Abstract

Physically based rendering에 대해 적는다. PBR은 빛이 사물과
상호작용하는 것을 현실셰계와 유사하게 표현하는 방법이다. PBR을
이해하려면 먼저 [lighting](/lighting/README.md)에 대해 먼저 이해해야
한다.

# Materials

* [VOLUME 1: THE THEORY OF PHYSICALLY BASED RENDERING AND SHADING](https://academy.allegorithmic.com/courses/b6377358ad36c444f45e2deaa0626e65)
  * 개발자 입장에서 본 PBR 요약
* [VOLUME 2: PRACTICAL GUIDELINES FOR CREATING PBR TEXTURES](https://academy.allegorithmic.com/courses/05171e19aa8dc19421385fd9bb9e016e)
  * 디자이너 입장에서 본 PBR 요약
* [Real Shading in Unreal Engine 4](http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf)
* [Physically Based Rendering Algorithms: A Comprehensive Study In Unity3D](http://www.jordanstevenstechart.com/physically-based-rendering)
* [BASIC THEORY OF PHYSICALLY-BASED RENDERING](https://www.marmoset.co/posts/basic-theory-of-physically-based-rendering/)
* [PBR이란무엇인가 @ tistory](http://lifeisforu.tistory.com/366)

# Workflow

* Metal / Roughness 와 Specualr / Glossniess와 같이 두가지 작업방식이
  존재한다.
* Metal / Roughness 는 다음과 같이 6가지 texture를 이용한다.
  * Base Color, Roughness, Metralic
  * Ambient Occlusion, Normal, Height
* Specular / Glossiness는 다음과 같이 6가지 texture를 이용한다.
  * Diffuse(Albedo), Glossiness, Specular
  * Ambient Occlusion, Normal, Height
* Metal / Roughness 와 Specular / Glossiness 는 각각 작업방식도
  다르지만 PBR을 구현하는 방식도 다르다.

# Key Elements

PBR은 다음과 같은 주요 요소들로 실현 된다.

* Light Rays
* Absorption and Scattering - Transparency and Translucency
* Diffuse and Specular Reflection - Microfacet Theory
* Color
* BRDF
* Energy Conservation (에너지 보존법칙)
* Fresnel Effect - F0 (Fresnel Reflectance at 0 Degrees)
* Conductors and Insulators - Metals and Non Metal
* Linear Space Rendering

# Light Rays



# Absorption and Scattering - Transparency and Translucency



# Diffuse and Specular Reflection - Microfacet Theory

# Color

# BRDF

# Energy Conservation (에너지 보존법칙)

# Fresnel Effect - F0 (Fresnel Reflectance at 0 Degrees)

빛이 피사체를 비출 때 그 피사체의 반사된 빛의 양은 그 피사체를
바라보는 각도에 따라 다른 현상을 프레넬 효과라고 한다. 

예를 들어 빛이 호수를 비추는 경우를 생각해 보자. 호수를 바라보는
각도와 물의 표면이 이루는 각이 수직일 때 호수의 바닥을 볼 수 있지만
호수를 바라보는 각도와 물의 표면이 이루는 각이 10도 미만일 때 즉
비스듬히 바라볼 때 호수의 바닥은 볼 수 없고 호수에 반사된 풍경이
보인다. 비스듬히 바라볼 때 반사된 빛의 양이 더욱 많다.

# Conductors and Insulators - Metals and Non Metal

# Linear Space Rendering

# Implementation

## Unity3d

Standard Shader는 PBR을 지원한다.  UnityStandardBRDF.cginc에서 다양한
근사와 조정이 이루어진 BRDF를 확인할 수
있다. UnityStandardCore.cginc와 UnityGlobalIllumination.cginc에서
이용법을 확인 할 수 있다. Standard Shader는 다음과 같이 플래폼별로
구현이 나뉘어져 있다.

- PC/console : 디즈니의 토런스 스패로(Torrance-Sparrow)반사 모델
- opengles 3.0이상 : 간략화된 쿡 토렌스 모델, 정점 조명을 이용한 확산/환경광
- opengles 2.0 : 정규화 블린퐁 반사 농도 함수 (reflection density function, RDF)와
  사전 계산 결과 텍스처 참조(lookup texture, LUT)

## Unrealengine

