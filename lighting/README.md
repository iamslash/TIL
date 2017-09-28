# Comments

다음의 용어들을 정리해본다.

hemispherical lighting model
image based lighting
irradiance environment map
image based reflection
image based refaction
image based fresnel
BRDF

# Abstract

- 3d그래픽의 라이팅에 대해 기술한다.

# Contents

* [Fundamentals](#fundamentals)

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

![](img/Light_dispersion_conceptual_waves350px.gif)

빛은 여러가지 전자기파들로 구성되어 있기 때문에 프리즘을 사용해서 각각을 분리할 수 있다.
빨간색 파장의 길이가 가장 길고 보라색 파장의 길이가 가장 짧다.

* [PBR 이란 무엇인가 - 1. 인간과 빛 @ tistory](http://lifeisforu.tistory.com/366)
* [Physically based rendering @ wikipedia](https://en.wikipedia.org/wiki/Physically_based_rendering)
* [BASIC THEORY OF PHYSICALLY-BASED RENDERING](https://www.marmoset.co/posts/basic-theory-of-physically-based-rendering/)
* [빛이란 무엇인가? - Barry R. Masters](http://e-ico.org/sites/default/files/pdfs/whatIsLightKorean.pdf)
* [Light @ wikipedia](https://en.wikipedia.org/wiki/Light)
* [파동의표시](http://lovescience.pe.kr/ms/chapter12/12_1_study2.htm)

## Eye

![인간의 눈 구조](img/HumanEye.png)

우리 눈의 망막(retina)를 확대해 보면 원추세포(cone cell: blue cone,
red cone, green cone)와 간상세포(rod)가 있다. 원추세포(cone cell)는
색상을 감지하고 간상세포(rod)는 명암을 감지한다.  원추세포는 그것을
구성하고 있는 감광세포의 종류에 따라 받아들일 수 있는 색상이 정해져
있다. 우측 그림을 보면 3종류의 원추세포가 존재하고 그것은 RGB와 같다.
이것은 텍스처를 제작할때 RGB채널을 사용하는 이유기도 하다.

우리 눈의 원추세포(cone cell)는 약 600만개이고 간상세포(rod)는 약
9000만개이다.  따라서 우리 눈은 색상 대비보다 명도 대비에 더욱
민감하다. 새의 눈은 인간 보다 많은 감광 색소를 가지고 있어서 자외선
까지 볼 수 있다고 한다.

* [PBR 이란 무엇인가 - 1. 인간과 빛 @ tistory](http://lifeisforu.tistory.com/366)
* [Color Vision in Birds](http://www.webexhibits.org/causesofcolor/17B.html)
* [원추세포 @ 위키피디아](https://ko.wikipedia.org/wiki/%EC%9B%90%EC%B6%94%EC%84%B8%ED%8F%AC)

## 조도와 휘도 (illuminance & luminance)

* []()
* []()
* []()

## 빛의 감쇠 (attenuation)

## 광원의 밝기 - 광속 (luminous flux)

## 조도 측정

## 휘도 측정

# Lambert's cosine law

확산반사(diffuse reflectance)가 일어나는 표면의 한 점에서의
복사강도(radiant intensity)I는, 입사광의 단위벡터 L과 표면의 법선
벡터인 면법선 (surface normal)N이 이루는 각도 θ의 코사인에 비례한다.

# Lambertian Reflectance Model

(Johann Heinrich Lambert)[https://en.wikipedia.org/wiki/Johann_Heinrich_Lambert]가 
1760년에 그의 저서 [Photometria](https://en.wikipedia.org/wiki/Photometria)에서 제안한
lighting model이다. lambert's cosine law를 기본으로 하며 diffuse relection을 구하는 방법은
다음과 같다.

![](img/lambert_reflectance_model.png)

![](img/lambert_reflectance_model_eq.png)

```latex
\begin{align*}
I_{D} &= \text{diffuse reflectance} \\
L     &= \text{normalized light direction vector} \\
N     &= \text{surface's normal vector} \\
C     &= \text{color} \\
I_{L} &= \text{intenciry of the incomming light} \\
\vspace{5mm}
I_{D} = L \cdot N C I_{L} \\
\end{align*}
```

다음은 lambertian reflectance model을 unity3d shader lab으로 구현한 것이다.
[참고](https://github.com/ryukbk/mobile_game_math_unity)

```cpp
Shader "Custom/Diffuse" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
	}
	SubShader {
		Pass {
			Tags { "LightMode" = "ForwardBase" }
			
			GLSLPROGRAM
	        #include "UnityCG.glslinc"
	        #if !defined _Object2World
	        #define _Object2World unity_ObjectToWorld
	        #endif

	        uniform vec4 _LightColor0;

	        uniform vec4 _Color;
	         
	        #ifdef VERTEX
			out vec4 color;

	        void main() {	            
	            vec3 surfaceNormal = normalize(vec3(_Object2World * vec4(gl_Normal, 0.0)));
	            vec3 lightDirectionNormal = normalize(vec3(_WorldSpaceLightPos0));
	            vec3 diffuseReflection = vec3(_LightColor0) * vec3(_Color) * max(0.0, dot(surfaceNormal, lightDirectionNormal));
	            color = vec4(diffuseReflection, 1.0);
	            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	        }
	        #endif

	        #ifdef FRAGMENT
			in vec4 color;

	        void main() {
	           gl_FragColor = color;
	        }
	        #endif

	        ENDGLSL
         }
	} 
	//FallBack "Diffuse"
}
```

# Half Lambert Diffuse

[Half lambert](https://developer.valvesoftware.com/wiki/Half_Lambert)
는 half-life라는 게임에서 처음 등장한 기술이다. 앞서 살펴 본
lambertian reflectance model은 어두운 부분이 너무 어둡기 때문에 이것을
보완 하고자 N과 L의 내적값을 [-1,1]에서 [0,1]로 조정한 것이다.

![](File-Alyx_lambert_half_lambert.jpg)

![](img/File-Lambert_vs_halflambert.png)


다음은 half lambert diffuse를 unity3d shaderlab으로 구현한 것이다.
[참고](https://github.com/ryukbk/mobile_game_math_unity)

```cpp
Shader "Custom/Half Lambert" {
	Properties {
		_Color ("Color", Color) = (1,1,1,1)
	}
	SubShader {
		Pass {
			Tags { "LightMode" = "ForwardBase" }
			
			GLSLPROGRAM
	        #include "UnityCG.glslinc"
	        #if !defined _Object2World
	        #define _Object2World unity_ObjectToWorld
	        #endif

	        uniform vec4 _LightColor0;

	        uniform vec4 _Color;
	         
	        #ifdef VERTEX
	        out vec4 color;

	        void main() {	            
	            vec3 surfaceNormal = normalize(vec3(_Object2World * vec4(gl_Normal, 0.0)));
	            vec3 lightDirectionNormal = normalize(vec3(_WorldSpaceLightPos0));
	            float halfLambert = max(0.0, dot(surfaceNormal, lightDirectionNormal)) * 0.5 + 0.5;
	            vec3 diffuseReflection = vec3(_LightColor0) * vec3(_Color) * halfLambert * halfLambert;
	            color = vec4(diffuseReflection, 1.0);
	            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	        }
	        #endif

	        #ifdef FRAGMENT
	        in vec4 color;

	        void main() {
	           gl_FragColor = color;
	        }
	        #endif

	        ENDGLSL
         }
	} 
	//FallBack "Diffuse"
}
```

# Phong Reflectance  Model

[Bui Tuong Phone](https://en.wikipedia.org/wiki/Bui_Tuong_Phong)
이 1975년에 제안한 lighting model이다.

phong reflection은 ambient, diffuse, specular term의 합으로 구한다.

![](img/Phong_components_version_4.png)

![](img/phong_reflectance_model.png)

![](img/phong_reflectance_model_eq.png)

```latex
\begin{align*}
I_{P} &= \text{phong reflectance} \\
I_{A} &= \text{ambient term} \\
I_{D} &= \text{diffuse term} \\
I_{S} &= \text{specular term} \\
\vspace{5mm}
L     &= \text{normalized light direction vector} \\
N     &= \text{surface's normal vector} \\
C     &= \text{color} \\
I_{L} &= \text{intenciry of the incomming light} \\
R     &= \text{reflected ray of light} \\
V     &= \text{normalized vector toward the viewpoint} \\
H     &= \text{normlized vector that is halfway between V and L} \\
P     &= \text{vecotr obtained by orthogonal projection of R to N} \\
A     &= \text{ambient light} \\
α     &= \text{shiness} \\
\vspace{5mm}
P     &= N(L \cdot N)
R - P &= P - L
R     &= 2P - L
      &= 2N(L \cdot N) - L
\vspace{5mm}
I_{A} &= A C\\
I_{D} &= L \cdot N C I_{L} \\
I_{S} &= I_{L}C(max(0, R \cdot V))^{α}\\
\end{align*}
```

위의 식에서 R을 구하는데 내적연산을 사용한다. 내적은 계산 비용이 많기 때문에
R대신 H를 이용해서 같은 효과를 얻을 수 있다. 이것을 Blinn-Phong reflection model
이라고 한다.

```latex
H = \frace{L + V}{|L+V|}
```

![](img\File-Blinn_phong_comparison.png)

# Gouraud shading

phong reflectance model을 vertex shader에 적용한 것

다음은 gouraud shading을 unity3d shader lab으로 구현한 것이다.
[참고](https://github.com/ryukbk/mobile_game_math_unity)

```cpp
Shader "Custom/Gouraud" {
	Properties {
		_Color ("Diffuse Color", Color) = (1,1,1,1)
		_SpecularColor ("Specular Color", Color) = (1,1,1,1)
		_SpecularExponent ("Specular Exponent", Float) = 3
	}
	SubShader {
		Pass {
			Tags { "LightMode" = "ForwardBase" }
			
			GLSLPROGRAM
	        #include "UnityCG.glslinc"
	        #if !defined _Object2World
	        #define _Object2World unity_ObjectToWorld
	        #endif

	        uniform vec4 _LightColor0;

	        uniform vec4 _Color;
	        uniform vec4 _SpecularColor;
	        uniform float _SpecularExponent;

	        #ifdef VERTEX
			out vec4 color;

	        void main() {
	            vec3 ambientLight = gl_LightModel.ambient.xyz * vec3(_Color);

	            vec3 surfaceNormal = normalize((_Object2World * vec4(gl_Normal, 0.0)).xyz);
	            vec3 lightDirectionNormal = normalize(_WorldSpaceLightPos0.xyz);
	            vec3 diffuseReflection = _LightColor0.xyz * _Color.xyz * max(0.0, dot(surfaceNormal, lightDirectionNormal));

                vec3 viewDirectionNormal = normalize((vec4(_WorldSpaceCameraPos, 1.0) - _Object2World * gl_Vertex).xyz);
				vec3 specularReflection = _LightColor0.xyz * _SpecularColor.xyz
					* pow(max(0.0, dot(reflect(-lightDirectionNormal, surfaceNormal), viewDirectionNormal)), _SpecularExponent);              

	            color = vec4(ambientLight + diffuseReflection + specularReflection, 1.0);
	            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	        }
	        #endif

	        #ifdef FRAGMENT
	        in vec4 color;

	        void main() {
	           gl_FragColor = color;
	        }
	        #endif

	        ENDGLSL
         }
	} 
	//FallBack "Diffuse"
}
```

# Phong Shading

phong reflectance model을 fragment shader에 적용한 것

다음은 phong shading을 unity3d shader lab으로 구현한 것이다.
[참고](https://github.com/ryukbk/mobile_game_math_unity)

```cpp
Shader "Custom/Phong" {
	Properties {
		_Color ("Diffuse Color", Color) = (1,1,1,1)
		_SpecularColor ("Specular Color", Color) = (1,1,1,1)
		_SpecularExponent ("Specular Exponent", Float) = 3
	}
	SubShader {
		Pass {
			Tags { "LightMode" = "ForwardBase" }
			
			GLSLPROGRAM
	        #include "UnityCG.glslinc"
	        #if !defined _Object2World
	        #define _Object2World unity_ObjectToWorld
	        #endif

	        uniform vec4 _LightColor0;

	        uniform vec4 _Color;
	        uniform vec4 _SpecularColor;
	        uniform float _SpecularExponent;

	        #ifdef VERTEX
			out vec4 glVertexWorld;
			out vec3 surfaceNormal;

	        void main() {	            
	            surfaceNormal = normalize((_Object2World * vec4(gl_Normal, 0.0)).xyz);
	            glVertexWorld = _Object2World * gl_Vertex;

	            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	        }
	        #endif

	        #ifdef FRAGMENT
			in vec4 glVertexWorld;
			in vec3 surfaceNormal;

	        void main() {
	        	vec3 ambientLight = gl_LightModel.ambient.xyz * vec3(_Color);
	        
				vec3 lightDirectionNormal = normalize(_WorldSpaceLightPos0.xyz);
	            vec3 diffuseReflection = _LightColor0.xyz * _Color.xyz * max(0.0, dot(surfaceNormal, lightDirectionNormal));

                vec3 viewDirectionNormal = normalize((vec4(_WorldSpaceCameraPos, 1.0) - glVertexWorld).xyz);
				vec3 specularReflection = _LightColor0.xyz * _SpecularColor.xyz
					* pow(max(0.0, dot(reflect(-lightDirectionNormal, surfaceNormal), viewDirectionNormal)), _SpecularExponent);                      
	        
	        	gl_FragColor = vec4(ambientLight + diffuseReflection + specularReflection, 1.0);
	        }
	        #endif

	        ENDGLSL
         }
	} 
	//FallBack "Diffuse"
}
```

# Rim Lighting

빛에 의해 오브젝트의 외곽이 빛나는 현상. N과 L의 사이각이 0일때 가장
약하고 90일때 가장 강하다.

다음은 phong shading과 rim lighting을 unity3d shader lab으로 구현한 것이다.
[참고](https://github.com/ryukbk/mobile_game_math_unity)

```cpp
Shader "Custom/Rim" {
	Properties {
		_Color ("Diffuse Color", Color) = (1,1,1,1)
		_SpecularColor ("Specular Color", Color) = (1,1,1,1)
		_SpecularExponent ("Specular Exponent", Float) = 3
		_RimColor ("Rim Color", Color) = (0,1,0,1)
	}
	SubShader {
		Pass {
			Tags { "LightMode" = "ForwardBase" }
			
			GLSLPROGRAM
	        #include "UnityCG.glslinc"
	        #if !defined _Object2World
	        #define _Object2World unity_ObjectToWorld
	        #endif

	        uniform vec4 _LightColor0;

	        uniform vec4 _Color;
	        uniform vec4 _SpecularColor;
	        uniform float _SpecularExponent;
	        uniform vec4 _RimColor;

	        #ifdef VERTEX
	        out vec4 glVertexWorld;
	        out vec3 surfaceNormal;

	        void main() {	            
	            surfaceNormal = normalize((_Object2World * vec4(gl_Normal, 0.0)).xyz);
	            glVertexWorld = _Object2World * gl_Vertex;

	            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	        }
	        #endif

	        #ifdef FRAGMENT
	        in vec4 glVertexWorld;
	        in vec3 surfaceNormal;

	        void main() {
	        	vec3 ambientLight = gl_LightModel.ambient.xyz * vec3(_Color);
	        
				vec3 lightDirectionNormal = normalize(_WorldSpaceLightPos0.xyz);
	            vec3 diffuseReflection = _LightColor0.xyz * _Color.xyz * max(0.0, dot(surfaceNormal, lightDirectionNormal));

                vec3 viewDirectionNormal = normalize((vec4(_WorldSpaceCameraPos, 1.0) - glVertexWorld).xyz);
				vec3 specularReflection = _LightColor0.xyz * _SpecularColor.xyz
					* pow(max(0.0, dot(reflect(-lightDirectionNormal, surfaceNormal), viewDirectionNormal)), _SpecularExponent);                      

	        	float rim = 1.0 - saturate(dot(viewDirectionNormal, surfaceNormal));        
	        	gl_FragColor.xyz = ambientLight + diffuseReflection + specularReflection + vec3(smoothstep(0.5, 1.0, rim)) * _RimColor.xyz;
	        	gl_FragColor.w = 1.0;
	        }
	        #endif

	        ENDGLSL
         }
	} 
	//FallBack "Diffuse"
}
```


# Cook-Torrance Model

다음은 cook-torrance model을 unity3d shader lab으로 구현한 것이다.
[참고](https://github.com/ryukbk/mobile_game_math_unity)

```cpp
Shader "Custom/CookTorrance" {
	Properties {
		_Color ("Diffuse Color", Color) = (1,1,1,1)
		_Roughness ("Roughness", Float) = 0.5
		_FresnelReflectance ("Fresnel Reflectance", Float) = 0.5
	}
	SubShader {
		Pass {
			Tags { "LightMode" = "ForwardBase" }
			
			GLSLPROGRAM
	        #include "UnityCG.glslinc"
	        #if !defined _Object2World
	        #define _Object2World unity_ObjectToWorld
	        #endif

	        uniform vec4 _LightColor0;

	        uniform vec4 _Color;
	        uniform float _Roughness;
	        uniform float _FresnelReflectance;

	        #ifdef VERTEX
	        out vec4 glVertexWorld;
	        out vec3 surfaceNormal;

	        void main() {	            
	            surfaceNormal = normalize((_Object2World * vec4(gl_Normal, 0.0)).xyz);
	            glVertexWorld = _Object2World * gl_Vertex;

	            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
	        }
	        #endif

	        #ifdef FRAGMENT
	        in vec4 glVertexWorld;
	        in vec3 surfaceNormal;

	        void main() {
	        	vec3 ambientLight = gl_LightModel.ambient.xyz * vec3(_Color);
	        
				vec3 lightDirectionNormal = normalize(_WorldSpaceLightPos0.xyz);
				float NdotL = saturate(dot(surfaceNormal, lightDirectionNormal));

			   	vec3 viewDirectionNormal = normalize((vec4(_WorldSpaceCameraPos, 1.0) - glVertexWorld).xyz);
			   	float NdotV = saturate(dot(surfaceNormal, viewDirectionNormal));

			    vec3 halfVector = normalize(lightDirectionNormal + viewDirectionNormal);
			    float NdotH = saturate(dot(surfaceNormal, halfVector));
			    float VdotH = saturate(dot(viewDirectionNormal, halfVector));

				float roughness = saturate(_Roughness);
			    float alpha = roughness * roughness;
			    float alpha2 = alpha * alpha;
				float t = ((NdotH * NdotH) * (alpha2 - 1.0) + 1.0);
				float PI = 3.1415926535897;
				float D = alpha2 / (PI * t * t);

			    float F0 = saturate(_FresnelReflectance);
			    float F = pow(1.0 - VdotH, 5.0);
			    F *= (1.0 - F0);
			    F += F0;

			    float NH2 = 2.0 * NdotH;
			    float g1 = (NH2 * NdotV) / VdotH;
			    float g2 = (NH2 * NdotL) / VdotH;
			    float G = min(1.0, min(g1, g2));

			    float specularReflection = (D * F * G) / (4.0 * NdotV * NdotL + 0.000001);
				vec3 diffuseReflection = _LightColor0.xyz * _Color.xyz * NdotL;

			    gl_FragColor = vec4(ambientLight + diffuseReflection + specularReflection, 1.0);
	        }
	        #endif

	        ENDGLSL
         }
	} 
	//FallBack "Diffuse"
}
```

# Oren-Nayar Model

# [PBR](../pbr/README.md)

# Ray Traciing

# Radiosity
