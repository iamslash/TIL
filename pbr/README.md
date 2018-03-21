<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Materials](#materials)
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

Physically based rendering에 대해 적는다. PBR을 이해하려면
먼저 [lighting](/lighting/README.md)에 대해 알아야 한다.

# Materials

* [VOLUME 1: THE THEORY OF PHYSICALLY BASED RENDERING AND SHADING](https://academy.allegorithmic.com/courses/b6377358ad36c444f45e2deaa0626e65)
  * 개발자 입장에서 본 PBR 요약
* [VOLUME 2: PRACTICAL GUIDELINES FOR CREATING PBR TEXTURES](https://academy.allegorithmic.com/courses/05171e19aa8dc19421385fd9bb9e016e)
  * 디자이너 입장에서 본 PBR 요약
* [Physically Based Rendering Algorithms: A Comprehensive Study In Unity3D](http://www.jordanstevenstechart.com/physically-based-rendering)
* [BASIC THEORY OF PHYSICALLY-BASED RENDERING](https://www.marmoset.co/posts/basic-theory-of-physically-based-rendering/)
* [PBR이란무엇인가 @ tistory](http://lifeisforu.tistory.com/366)

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

