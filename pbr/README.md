- [Abstract](#abstract)
- [Materials](#materials)
- [Workflow](#workflow)
- [Key Elements](#key-elements)
- [Light Rays](#light-rays)
- [Absorption and Scattering - Transparency and Translucency](#absorption-and-scattering---transparency-and-translucency)
- [Diffuse and Specular Reflection](#diffuse-and-specular-reflection)
- [Microfacet Theory](#microfacet-theory)
- [Color](#color)
- [BRDF](#brdf)
- [Energy Conservation (에너지 보존법칙)](#energy-conservation-%EC%97%90%EB%84%88%EC%A7%80-%EB%B3%B4%EC%A1%B4%EB%B2%95%EC%B9%99)
- [Fresnel Effect - F0 (Fresnel Reflectance at 0 Degrees)](#fresnel-effect---f0-fresnel-reflectance-at-0-degrees)
- [Conductors and Insulators](#conductors-and-insulators)
  - [Metals](#metals)
  - [Non Metal](#non-metal)
- [Linear Space Rendering](#linear-space-rendering)
- [Implementation](#implementation)
  - [WegGL](#weggl)
  - [Unity3d](#unity3d)
    - [Unity3d builtin](#unity3d-builtin)
  - [Unreal Engine 4](#unreal-engine-4)

-------------------------------------------------------------------------------

# Abstract

Physically based rendering 에 대해 적는다. PBR은 빛이 사물과
상호작용하는 것을 현실셰계와 유사하게 표현하는 방법이다. PBR 을
이해하려면 먼저 [lighting](/lighting/README.md)에 대해 이해해야
한다.

PBR을 구현하는 방법은 다양하다. 성능을 고려하기 위해 최적화도 해야 하고
아티스트 입장에서 원하는 효과가 다르기 때문에 사용된 수식도 다를 수 있다.
PBR을 추상적으로 먼저 이해하고 엔진 별로 구현한 예를 살펴보자.

# Materials

* [물리기반 렌더링의 최근 기술동향과 이슈 - 진성아](http://www.sersc.org/journals/AJMAHS/vol6_no9_2016/6.pdf)
  * PBR 한글 논문
* [Physically based rendering](https://bib.irb.hr/datoteka/890912.Final_0036473349_56.pdf)
  * PBR 영어 논문
* [VOLUME 1: THE THEORY OF PHYSICALLY BASED RENDERING AND SHADING](https://academy.allegorithmic.com/courses/the-pbr-guide-part-1)
  * 개발자 입장에서 본 PBR 요약
* [VOLUME 2: PRACTICAL GUIDELINES FOR CREATING PBR TEXTURES](https://academy.allegorithmic.com/courses/the-pbr-guide-part-2)
  * 디자이너 입장에서 본 PBR 요약
* [Real Shading in Unreal Engine 4](https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf)
  * 2013 SIGRAPH paper
  * [번역](http://lifeisforu.tistory.com/348)
* [Physically Based Rendering Algorithms: A Comprehensive Study In Unity3D](http://www.jordanstevenstechart.com/physically-based-rendering)
  * unity3d를 이용하여 PBR을 자세히 설명했다.
* [BASIC THEORY OF PHYSICALLY-BASED RENDERING](https://www.marmoset.co/posts/basic-theory-of-physically-based-rendering/)
* [PBR이란무엇인가 @ tistory](http://lifeisforu.tistory.com/366)
* [Physically Based Rendering](https://pbrt.org/)
  * PBR 바이블
  * [src](https://github.com/mmp/pbrt-v3/)

# Workflow

`Metal / Roughness` 와 `Specualr / Glossniess`와 같이 두가지 작업방식이 존재한다.
`Metal / Roughness` 는 `Base Color, Roughness, Metallic, Ambient Occlusion, Normal, Height `과 같이 6 가지 texture 를 이용한다. `Specular / Glossiness` 는 `Diffuse(Albedo), Glossiness, Specular, Ambient Occlusion, Normal, Height` 와 같이 6가지 texture 를 이용한다. `Metal / Roughness` 와 `Specular / Glossiness`는 각각 작업방식과 구현하는 방식이 다를 뿐 결과물은 유사하다.

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

![](https://cdn.allegorithmic.com/images/academy/fe2fcf01-5912-4975-9e72-f94277eca7a7)

PBR을 구현하려면 가장 먼저 빛의 성질에 대해 이해해야 한다. 빛은 광원을 떠나서 피사체의 표면에 부딛힌다. 표면으로 들어오는 빛은 **입사광선(incident ray)**, 표면에서 반사되어
나가는 빛은 **반사광선(reflected ray)**, 표면을 통과하여 나가는 광선은 **굴절광선(refracted ray)** 라고 한다. 이때 일부 광선은 표면에서 **흡수(absorbed)** 되거나 **산란(scattered)** 되기도 한다.

# Absorption and Scattering - Transparency and Translucency

빛이 피사체의 표면에 부딛히면 일부 광선은 표면에서 **흡수(absorbed)** 되거나 **산란(scattered)** 되기도 한다. 

흡수된 빛은 밝기가 줄어들고 줄어든 만큼의 빛은 열 에너지로 바뀐다. 흡수되고 남은 빛은 파장에 따라 색깔이 변화하지만 방향이 변하지는 않는다.

산란된 빛은 밝기가 줄어들지 않지만 방향이 랜덤하게 바뀐다. 방향이 바뀌는 정도는 피사체 표면의 재질에 따라 다르다.

만약 빛이 피사체의 표면에 부딛힐 때 흡수도 안되고 산란도 없다면 대부분의 빛은 투과될 것이다.
예를 들면 깨끗한 물의 수영장 속에서 눈을 떴을 때 사물들은 그대로 보인다. 그러나 더러운 물의 수영장 속에서 눈을 뜨면 그렇지 않다.

빛의 흡수 및 산란은 피사체의 두깨와 관련이 깊다. 만약 피사체의 두깨가 얼마인지 정보를 texture(thickness map) 에 저장한다면 흡수 및 산란의 정도를 조절할 수도 있다.

# Diffuse and Specular Reflection

![](https://cdn.allegorithmic.com/images/academy/e221d725-3d3e-448f-8637-339eda272186)

빛이 피사체의 표면에 부딛힐 때 일부광선은 반사된다고 언급했다. 반사된 빛은 정반사(specular reflection) 와 난반사(diffuse reflection) 로 나눌 수 있다.
피사체 표면이 균일 하면 정반사가 주로 발생되서 하이라이트가 작고 또렷하다. 피사체 표면이 균일 하지 않으면 때문에 정반사보다 난반사가 주로 발생되서 하이라이트가 넓고 흐리다.

Lambertian model 은 diffuse reflection 을 다룰 때 표면의 거칠기를 무시한다. 그러나 Oren-Nayar model 은 diffuse reflection 을 다룰 때 표면의 거칠기를 고려한다.

빛은 서로다른 매질을 지나갈 때 굴절된다. 이때 빛이 굴절되는 정도를 굴절률(Index of refraction) 이라고 하고 매질에 따라 값이 다르다.

# Microfacet Theory

빛은 피사체의 표면에 부딛힐때 일부는 반사된다. 반사되는 빛은 표면의 거칠기에 따라 정반사(specular reflection) 와 난반사(diffuse reflection) 로 나눠진다. 표면의 거칠기는 PBR 작업방식에 따라 roughness, smoothness, glossiness, micro-surface 등으로 달리 부른다. 표면의 거칠기는 특별한 texture(roughness map) 에 저장할 수 있다.

![](https://cdn.allegorithmic.com/images/academy/c1a54a83-66a5-49e0-9e0e-4df25a3fce48)

![](https://cdn.allegorithmic.com/images/academy/294726f5-0e48-42ec-979b-f4b6fe7dd618)

피사체의 표면은 작은 표면들이 서로 다른 각도를 이루고 모여 있다고 할 수 있다. 이것을 microfacet theory 라고 한다. PBR 의 BRDF 는 microfacet theory 에 기반을 두고 있다.

# Color

빛이 사과를 향해 나아간다고 해보자. 사과의 표면에 빛이 부딛히면 일부는 흡수되고 일부는 반사되어 우리 눈으로 들어온다. 앞서 언급한 것처럼 반사는 정반사와 난반사로 나눠진다. 빨간 파장의 난반사(diffuse refelction) 광선이 눈으로 들어오면 우리는 사과가 빨갛다고 생각할 수 있다. 이처럼 피사체의 색은 피사체의 표면이 난반사한 빛의 파장에 따라 결정된다.

정반사(specular refelction) 빛의 색은 난반사(diffuse reflection) 빛의 색처럼 빨갛지 않고
하얗다. 즉 광원의 색상과 같다. 이것은 사과의 표면이 부도체 (dieletrics) 이기 때문이다. 도체(conductors)와 부도체(insulators)의 설명은 [Conductors and Insulators](#conductors-and-insulators---metals-and-non-metal)에서 자세히 설명한다.

# BRDF

![](/lighting/img/BSDF05_800.png)

**BRDF (bidirectional reflectance distribution function)** 는 빛이 피사체의 표면을 비추고 반사가 발생할 때 얼만큼의 빛이 반사되는지 결정하는 함수이다.

# Energy Conservation (에너지 보존법칙)

반사된 빛의 양은 광원에서 출발한 빛의 양보다 크지 않다. 당연한
이야기다. 

# Fresnel Effect - F0 (Fresnel Reflectance at 0 Degrees)

빛이 피사체를 비출 때 그 피사체의 반사된 빛의 양은 그 피사체를
바라보는 각도에 따라 다른 현상을 프레넬 효과라고 한다. 각도가 0 일때
반사된 빛의 양을 `F0(Fresnel Zero)` 라고 하고 굴절된 빛의 양은
`1-F0` 라고 할 수 있다.

예를 들어 빛이 호수를 비추는 경우를 생각해 보자. 호수를 바라보는
각도와 물의 표면이 이루는 각이 수직일 때 호수의 바닥을 볼 수 있지만
호수를 바라보는 각도와 물의 표면이 이루는 각이 10도 미만일 때 즉
비스듬히 바라볼 때 호수의 바닥은 볼 수 없고 호수에 반사된 풍경이
보인다. 비스듬히 바라볼 때 반사된 빛의 양이 더욱 많기 때문이다.

# Conductors and Insulators

![](https://cdn.allegorithmic.com/images/academy/38d004ae-33fc-4175-8abe-6bd1b4fdaeb8)

위의 그림과 같이 피사체가 금속성에 가까운지 비금속성에 가까운지에 따라 F0(Frenel zero) 는 다르다. 

## Metals 

빛이 금속 피사체의 표면에 부딛히면 우리는 반짝임을 인식한다. 빛이 반짝인다는 것은 specular reflection 은 많고 diffuse reflection 은 적다는 것을 의미한다.

![](https://cdn.allegorithmic.com/images/academy/7f5bf828-8ac5-49fd-b040-0f205d1eeb3d)

보통 금속의 경우 incident ray의 60-70%는 반사되고 나머지는 흡수된다.

## Non Metal

빛이 비금속 피사체의 표면에 부딛히면 우리는 덜 반짝임을 인식한다. 빛이 덜 반짝인다는 것은
specular reflection 은 적고 diffuse reflection 은 많다는 것을 의미한다. diffuse reflection 의 색은 피사체의 albedo color 와 같다.

![](https://cdn.allegorithmic.com/images/academy/838b33e0-3736-4b75-a043-926682cc413f)

일반적인 비금속의 F0 는 2-5% 이고 linear color space 값은 0.017-0.067(40-75 sRGB) 이다. 보석을 제외하고 보통의 비금속들은 F0 가 4% 를 넘지는 않을 것이다.

# Linear Space Rendering

인간의 눈은 밝은 것보다 어두운 것에 민감하다. RGB정보를 24bit 에 저장한다고 해보자. 어두운 정보를 저장할 때 많은 bit를 사용하고 밝은 정보를 저장할 때 적은 bit 를 사용한다면 bit 를 효율적으로 이용할 수 있다. 이렇게 부화하는 것을 gamma encoding 이라고 하고 이때의 색공간을 gamma color space 라고 한다. 관련된 표준으로 [sRGB](https://namu.wiki/w/sRGB)가 있다. 이미지를 저장할 때 주로 sRGB 로 저장한다. 모니터는 sRGB 이미지를 gamma decoding 하여 보여준다.

색은 linear color space 에서 계산하는 것이 gamma color space 에서 계산하는 것보다 정확하다.
shader 에서 sRGB(gamma color space) 로 저장된 이미지는 계산하기 전에 linear color space 로
변환해야 한다.

다음은 [glTF-WebGL-PBR/shaders/pbr-frag.glsl](https://github.com/KhronosGroup/glTF-WebGL-PBR/blob/master/shaders/pbr-frag.glsl) 에서 sRBG를 linear color space로 바꾸는 함수를 발췌한 것이다.

```glsl
vec4 SRGBtoLINEAR(vec4 srgbIn)
{
#ifdef MANUAL_SRGB

  #ifdef SRGB_FAST_APPROXIMATION
    vec3 linOut = pow(srgbIn.xyz, vec3(2.2));

  #else //SRGB_FAST_APPROXIMATION
    vec3 bLess = step(vec3(0.04045), srgbIn.xyz);
    vec3 linOut = mix(srgbIn.xyz/vec3(12.92), 
      pow((srgbIn.xyz+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );

  #endif //SRGB_FAST_APPROXIMATION
    return vec4(linOut,srgbIn.w);;

#else //MANUAL_SRGB
    return srgbIn;

#endif //MANUAL_SRGB
}
```

# Implementation

## WegGL

khronos group에서 제작한 wegl예제를 참고하자. [src](https://github.com/KhronosGroup/glTF-WebGL-PBR/blob/master/shaders/pbr-frag.glsl) 를 살펴보면 어떤 식을 사용해서 구현했는지 알 수 있다.

* [demo](http://github.khronos.org/glTF-WebGL-PBR/) 
* [src](https://github.com/KhronosGroup/glTF-WebGL-PBR)

## Unity3d

### Unity3d builtin 

unity의 Standard Shader는 PBR을 지원한다.  UnityStandardBRDF.cginc 에서 다양한
근사와 조정이 이루어진 BRDF를 확인할 수 있다. UnityStandardCore.cginc 와 UnityGlobalIllumination.cginc 에서 이용법을 확인 할 수 있다. Standard Shader는 다음과 같이 플래폼별로 구현이 나뉘어져 있다.

- PC/console : 디즈니의 토런스 스패로(Torrance-Sparrow)반사 모델
- opengles 3.0이상 : 간략화된 쿡 토렌스 모델, 정점 조명을 이용한 확산/환경광
- opengles 2.0 : 정규화 블린퐁 반사 농도 함수 (reflection density function, RDF)와
  사전 계산 결과 텍스처 참조(lookup texture, LUT)

[이곳](http://www.jordanstevenstechart.com/physically-based-rendering) 은 unity3d 를
이용하여 pbr 구현 과정을 자세히 풀어쓴 글이다.

## Unreal Engine 4

...