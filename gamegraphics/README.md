- [Abstract](#abstract)
- [Essentials](#essentials)
- [Materials](#materials)
- [Snippets](#snippets)
- [3D graphics api](#3d-graphics-api)
- [Opensource Game Engines](#opensource-game-engines)
- [Math Prerequisites](#math-prerequisites)
  - [Law Of Cosines](#law-of-cosines)
  - [Trigonometric Addition formulas](#trigonometric-addition-formulas)
  - [Half-Angle Formulas](#half-angle-formulas)
  - [Dot Product](#dot-product)
  - [Cross Product](#cross-product)
  - [Affine Transform](#affine-transform)
  - [affine space](#affine-space)
  - [affince space operation](#affince-space-operation)
  - [homogeneous coordinates (동차좌표)](#homogeneous-coordinates-동차좌표)
  - [tangent space](#tangent-space)
- [Rendering Pipeline](#rendering-pipeline)
- [Color Space](#color-space)
- [Polygon Mesh](#polygon-mesh)
- [Vertex Processing](#vertex-processing)
- [Rasterization](#rasterization)
- [Fragment Processing](#fragment-processing)
- [Output Merging](#output-merging)
- [Lighting](#lighting)
- [Shader Models](#shader-models)
- [Shader Language](#shader-language)
- [Curve](#curve)
- [Bump Mapping](#bump-mapping)
- [Environment Mapping](#environment-mapping)
- [Light Mapping](#light-mapping)
- [Shadow Mapping](#shadow-mapping)
- [Ambient Occlusion Mapping](#ambient-occlusion-mapping)
- [Deferred Shading](#deferred-shading)
- [Animation](#animation)
- [Collision Detection](#collision-detection)

-------------------------------------------------------------------------------

# Abstract

게임 그래픽스에 대해 정리한다.

# Essentials

- [Introduction to 3D Game Programming with Direct3D](http://www.d3dcoder.net/d3d12.htm)
  - frank luna의 명저
  - [src](https://github.com/d3dcoder/d3d12book)
- [3차원 그래픽스(게임 프로그래밍을위한)](http://media.korea.ac.kr/book/)
  - 3차원 그래픽스 기반이론을 매우 자세히 풀어썼다. 저자의
    홈페이지에서 제공하는 슬라이드는 각종 그림과 수식을 가득 포함하고 있다.
  - [3D Graphics for Game Programming lecture notes](3dgraphics_for_game_programming_lecture_notes/)
- [Real-Time Rendering](https://www.amazon.com/Real-Time-Rendering-Third-Edition-Akenine-Moller/dp/1568814240)
  - 기반이론이 1000페이지 넘게 잘 정리된 책이다.
- [Interactive 3D Graphics @ udacity](https://classroom.udacity.com/courses/cs291)
  - [Real-Time Rendering](https://www.amazon.com/Real-Time-Rendering-Third-Edition-Akenine-Moller/dp/1568814240)의 동영상 강좌
  - [syllabus](https://www.udacity.com/wiki/cs291/syllabus)
  - [comments](https://www.udacity.com/wiki/cs291/instructor-comments)
  - [wiki](https://www.udacity.com/wiki/cs291)
  - [three.js tutorial](http://stemkoski.github.io/Three.js/)
* [Physically Based Rendering](https://pbrt.org/)
  * PBR 바이블
  * [src](https://github.com/mmp/pbrt-v3/)
- [nvidia cg tutorial](http://http.developer.nvidia.com/CgTutorial/cg_tutorial_chapter01.html)
  - 컴퓨터그래픽스 기반 이론을 cg와 함께 설명한다.
- [game engine architecture](https://www.gameenginebook.com/)
  - 게임엔진구조를 다룬 책이다. 그러나 구현체가 없어서 아쉽다. [ogre3d](https://www.ogre3d.org/) 를 이용하여 공부해 보자.
- [shader development using unity5](http://shaderdev.com/p/shader-development-using-unity-5)
  - 유료이긴 하지만 가장 자세히 설명하는 동영상 강좌이다. 174$
- [Learn Opengl](https://learnopengl.com/)
  - very good opengl tutorial
  - VAO, VBO, EBO 를 그림으로 이해할 수 있었다.
  - [src](https://github.com/JoeyDeVries/LearnOpenGL)
- [OpenGL Programming Guide: The Official Guide to Learning OpenGL, Version 4.3](http://www.opengl-redbook.com/)
  - opengl red book
  - [src](https://github.com/openglredbook/examples)
- [OpenGL Superbible: Comprehensive Tutorial and Reference](http://www.openglsuperbible.com)
  - opengl blue book
  - [src](https://github.com/openglsuperbible/sb7code)
- [OpenGL Shading Language](https://www.amazon.com/OpenGL-Shading-Language-Randi-Rost/dp/0321637631/ref=sr_1_1?ie=UTF8&qid=1538565859&sr=8-1&keywords=opengl+shading+language)
  - opengl orange book 
- [unity3d manual](https://docs.unity3d.com/Manual/index.html) [unity3d tutorial](https://unity3d.com/kr/learn/tutorials)
  - unity3d manual과 tutorial이야 말로 잘 만들어진 엔진을 이용하여 computer graphcis로 입문 할 수 있는 좋은 교재이다. unity3d에서 제공하는 기능들을 위주로 학습한다.
- [unrealengine manual](https://docs.unrealengine.com/latest/KOR/index.html)
  - unrealengine manual역시 잘 만들어진 엔진을 이용하여 computer graphcis로 입문 할 수 있는
    좋은 교재이다.
- [webgl fundamentals](https://webglfundamentals.org/)
  - 게임그래픽스이론을 webgl에서 간단히 실습해 보자.
- [GPU gems](https://developer.nvidia.com/gpugems/GPUGems/gpugems_pref01.html)
  - nvidia 에서 발간하는 advanced graphics 무료 책
  - [src](https://github.com/QianMo/GPU-Gems-Book-Source-Code)
- [modern opengl tutorial](http://ogldev.atspace.co.uk/)
  - 40여개의 튜토리얼이 단계별로 잘 설명되어 있다.
  - [src](http://ogldev.atspace.co.uk/ogldev-source.zip)
- [opengl-tutorial](http://www.opengl-tutorial.org/)
  - 초급, 중급의 튜토리얼이 설명되어 있다. particle 에 대한 글이 있음.
  - [src](https://github.com/opengl-tutorials/ogl)

# Materials

- [이득우의 게임 기술 블로그 @ naver](https://m.blog.naver.com/PostList.nhn?blogId=destiny9720&categoryNo=22&listStyle=style1)
  - 이득우님의 게임수학 블로그 
- [Unity3d tutorials](https://catlikecoding.com/unity/tutorials/)
  - water, rendering, noise, DOF, Bloom, FXAA, Triplanar Mapping 등등 많은 주제들을 다루는 상세한 블로그
- [유니티로 배우는 게임 수학](http://www.yes24.com/24/goods/30119802?scode=032&OzSrank=1)
  - 요약 설명이 많아서 초보자 보기에는 불편한 설명이다. 하지만 기반
    내용을 정리하는 용도로 좋다. 짐벌락, PBR에 대한 간략한 설명은 특히
    괜찮았다.
  - [src](https://github.com/ryukbk/mobile_game_math_unity)
- [pixar in a box @ khan](https://www.khanacademy.org/partner-content/pixar)
  - 픽사의 그래픽스 강좌
- [realtimerendering graphics books page](http://www.realtimerendering.com/books.html)
  - 아주 많은 책들이 정리되어 있다. 언제 다 보지?
- [awesome gamedev @ github](https://github.com/ellisonleao/magictools)
- [awesome graphics resources @ github](https://github.com/mattdesl/graphics-resources)
- [awesome computer vision @ github](https://github.com/jbhuang0604/awesome-computer-vision)
- [awesome opengl @ github](https://github.com/eug/awesome-opengl)
- [directx11 tutorials](http://www.rastertek.com/tutdx11.html)
  - 약 50여 개의 directx11 튜토리얼 
- [game engine development by Jamie King @ youtube](https://www.youtube.com/playlist?list=PLRwVmtr-pp04XomGtm-abzb-2M1xszjFx)
  - visual studio를 이용한 게임 엔진 개발
  - [src](https://github.com/1kingja/game-engine)
- [GPU how to work](http://pixeljetstream.blogspot.kr/2015/02/life-of-triangle-nvidias-logical.html)
  - GPU가 어떻게 작동하는 가를 렌더링 파이플 라인과 함께 설명함
- [Shadow algorithms for computer graphics](https://dl.acm.org/citation.cfm?id=563901)
  - shadow volume
- [casting curved shadows on curved surfaces](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.162.196&rep=rep1&type=pdf)
  - shadow mapping
- [Advanced Global Illumination, Second Edition](https://www.amazon.com/Advanced-Global-Illumination-Second-Philip/dp/1568813074)
- [Ke-Sen Huang's Home Page](http://kesen.realtimerendering.com/)
  - 컴퓨터그래픽스 컨퍼런스 자료 및 논문 모음
- [awesome graphics @ github](https://github.com/ericjang/awesome-graphics)
  - 컴퓨터그래픽스 논문등등 모음
- [digital lighting & rendering: third edition](http://www.3drender.com/light/)
  - 3D 라이팅과 렌더링 기법의 표준을 제시한 바이블을 만난다! 
- [opengl at songho](http://www.songho.ca/opengl/)
  - computer graphics의 이론들이 잘 정리되어 있다.
  - 특히 [OpenGL Normal Vector Transformation](http://www.songho.ca/opengl/gl_normaltransform.html)의 설명이 너무 좋았다.
- [Mathematics for 3D Game Programming and Computer Graphics, Third Edition](http://www.mathfor3dgameprogramming.com/)
  - 3D computer graphics를 위한 수학
- [GPG gems](http://www.satori.org/game-programming-gems/)
  - Game Programming Gems, 줄여서 GPG는 Nintendo of America의 수석 소프트웨어 엔지니어인 
    Mark Deloura가 시작한 게임 프로그래밍 서적 시리즈이다. 업계와 학계의 여러 저자들이 쓴 글들을 
    모은 일종의 앤솔로지 형태이다.
- [GPU pro](https://www.amazon.com/gp/product/149874253X?tag=realtimerenderin&pldnSite=1)
  - advand rendering technique
  - [src](https://github.com/wolfgangfengel/GPU-Pro-7)
- [GPU Zen](https://www.amazon.com/gp/product/B0711SD1DW?tag=realtimerenderin&pldnSite=1)
  - GPU pro의 후속작이다.
  - [src](https://github.com/wolfgangfengel/GPUZen)
- [ShaderX7](https://www.amazon.com/ShaderX7-Rendering-Techniques-Wolfgang-Engel/dp/1584505982)
- [scratchapixel 2.0](http://www.scratchapixel.com/)
  - computer graphics를 알기쉽게 설명한다.
- [intel developer zone game dev code samples](https://software.intel.com/en-us/gamedev/code-samples)
- [amd developer central tools & sdks](http://developer.amd.com/tools-and-sdks/graphics-development/)
- [nvidia gameworks](https://developer.nvidia.com/what-is-gameworks)
  - vulkan, opengl, directx sample
- [microsoft directx graphics samples](https://github.com/Microsoft/DirectX-Graphics-Samples)
  - 뭐지 이건

# Snippets

- [Introduction to 3D Game Programming with Direct3D src @ github](https://github.com/d3dcoder/d3d12book)
- [OpenGL Programming Guide: The Official Guide to Learning OpenGL, Version 4.3 src @ github](https://github.com/openglredbook/examples)
- [OpenGL Superbible: Comprehensive Tutorial and Reference src @ github](https://github.com/openglsuperbible/sb7code)

# 3D graphics api

* [DirectX](/directx/README.md)
* [OpenGL](/opengl/README.md)
* [webgl](https://webglfundamentals.org/)

# Opensource Game Engines

- [MiniEngine](https://github.com/Microsoft/DirectX-Graphics-Samples/tree/master/MiniEngine)
  - Microsoft 에서 DirectX 12 를 이용하여 만든 렌더링엔진
  - [DirectX 12: Demo engine: A Mini Engine Overview](https://www.youtube.com/watch?v=bggYcB1mFDI)
  - [DirectX 12: A MiniEngine Update](https://www.youtube.com/watch?v=bSTIsgiw7W0)
- [ogre3d](https://www.ogre3d.org/)
  - c++로 제작된 크로스플래폼 렌더링 엔진  
- [ogitor](https://github.com/OGRECave/ogitor)
  - qt 로 제작된 ogre3d scene builder 이다.
- [unrealengine 4 src @ github](https://github.com/EpicGames/UnrealEngine)
- [unity cs reference @ github](https://github.com/Unity-Technologies/UnityCsReference)
- [godot](https://godotengine.org/)
- [three.js](https://threejs.org/)
- [twgljs](https://twgljs.org/)
  - three.js 보다 가벼운 렌더링 엔진

# Math Prerequisites

## Law Of Cosines

![](img/Triangle_with_notations_2.svg.png)

```latex
c^{2} = a^{2} + b^{2} - 2ab\cos(\gamma)
```

![](img/cosinelaw.png)

## Trigonometric Addition formulas

[참고](http://mathworld.wolfram.com/TrigonometricAdditionFormulas.html)

## Half-Angle Formulas

[참고](http://mathworld.wolfram.com/Half-AngleFormulas.html)

## Dot Product

[참고](http://mathworld.wolfram.com/DotProduct.html)

## Cross Product

[참고](http://mathworld.wolfram.com/CrossProduct.html)

## Affine Transform

world transform, view transform 은 scaling, rotation, translation 등과
같이 기본적인 변환들을 조합하여 만들어진다. 한편 scaling, rotation 은
linear transform(선형변환) 의 범주에 속한다. translation(이동) 은
linear transform 에 속하지 않는다. 대신 linear transform 과 함께 affine
transform 의 범주에 속한다.

## affine space

vector space 에서는 vector 가 어디에 위치해 있던지 크기와 방향만 같다면
같은 vector 로 생각한다. vector space 에서 크기와 방향은 같지만 위치가
다른 vector 를 구분할 필요가 있다. 그래서 affine space 를 만들어냈다.
affine space 에서는 position 을 추가하여 vector 의 위치를 표현한다.

vector space 는 affine space 에 포함되고 affine space 는 projection
space 에 포함된다.

## affince space operation

vector 와 vector 의 `+, -` 는 vector 이다. scala 와 vector 의 `*, /` 는
vector 이다. vector 와 point 의 `+, -` 는 point 이다. point 와 point 의
`-` 는 vector이다. point 와 point 의 `+` 는 허용되지 않는다. (단 계수의 합이
1인 경우는 허용된다.)

![](img/affine_space_op.png)

affine space 에서 point A 는 point O 에서 point  A 로 가는 vector 로 
생각 할 수 있다. 따라서 C = A + 0.5 * (B - A) 이다.
point A 와 vector B - A 의 합은 point 임을 알 수 있다.

이때 0.5 대신 k 를 도입하여 다음과 같이 표기할 수 있다.

```
C = A + k(B - A) (0 <= k <= 1)
C = (1 - k)A + kB
```

k 가 1 이면 C = B 이고 k 가  0이면 C = A 이다. 이처럼 계수의 합이 1 인 경우는
point 와 point 의 덧셈 연산이 가능하고 이런 경우를 affine sum 이라고 한다.

## homogeneous coordinates (동차좌표)

n-tuple 에 하나의 차원 `w` 을 추가시켜서 vector 혹은 point 를 표현할 수 있는 좌표체계이다. 하나의 좌표체계로 vector 혹은 point 를 표현할 수 있기 때문에 하나의 수식으로 vector 와 point 의 연산을 표현할 수 있다. 예를 들어서 `v = (v1, v2)` 가 있다고 하자. `v` 의 homogeneous coordinates `v' = (v1, v2, w)` 이고 `w` 가 0 이면 vector 를 `w` 가 1 이면 point 를 의미한다. 

`w` 의 값이 `1` 보다 큰 homogeneous coordinates 의 경우 각 성분을 `w` 로 나누어 `x, y, z` 가 모두 같다면 같은 point 으로 취급한다. 따라서 다음과 같은 position 들은 모두 같다.  `(5, 1, 1) = (10, 2, 2) = (15, 3, 3) = (20, 4, 4)`

실제로 viewport transform 에서 normalized device coordiates 를 window space coordiates 로 변환할때 point 의 각 성분을 `w` 로 나누는 연산을 한다. viewport transform 이후에는 모든 점들의 `w` 가 1 이기 때문에 더이상 `w` 는 필요 없게 된다.

## tangent space

특정한 point 의 normal, tangent, binormal
vector 를 축으로 하는 공간이다. normal mapping 을 위해
vertex 의 normal 값을 기준으로 tangent space 를 표현하는 TBN
행렬을 구할 수 있고 normal map 에 저장된 단위 tangent space normal
vector 와 연산하여 최종 normal vector 를 구할 수 있다.

# Rendering Pipeline

[rendering pipeline](/renderingpipeline/README.md)

# Color Space

[Gamma Space vs Linear Space](/colorspace/README.md)

# Polygon Mesh

![](img/vertex_index_buffer.png)

위의 그림은 vertex buffer 와 index buffer 를 표현한 것이다.  polygon
t1 을 주목하자. index buffer 에 vertex buffer 의 index 가 CCW(counter
clock wise, 반시계방향) 으로 3 개 저장되어 있다.

![](img/surface_normal_ccw.png)

![](img/surface_normal_ccw_eq.png)

surface normal 은 중요하다. 위의 그림처럼 polygon 을 구성하는 vertex p1, p2, p3 에 
대해서 vector v1, v2 를 외적하고 정규화해서 surface normal 을 구한다.
반드시 p1, p2, p3 는 CCW 로 index buffer 에 저장되어 있어야 한다.

![](img/surface_normal_ccw.png)

![](img/surface_normal_ccw_eq.png)

만약 p1, p2, p3가 CW로 index buffer에 저장되어 있다면 surface normal은
반대 방향으로 만들어 진다.

![](img/vertex_normal.png)

![](img/vertex_normal_eq.png)

vertex normal 은 surface normal 보다 더 중요하다.
vertex normal 과 인접한 polygon 들의 surface normal 을 이용하여
구할 수 있다.

![](img/RHS_LHS.png)

![](img/RHS_LHS_normal.png)

좌표계는 오른손 좌표계와 왼손 좌표계가 있다. opengl 은 RHS 를
directx3D 는 LHS 를 사용한다. surface normal이 구의 바깥쪽으로
만들어질려면 RHS 의 경우 index buffer에 polygon을 구성하는 vertex 들의
index 가 CCW 로 저장되어야 하고 LHS 의 경우 CW 로 저장되어야 한다.

![](img/RHS_to_LHS_a.png)

![](img/RHS_to_LHS_b.png)

![](img/RHS_to_LHS_c.png)

RHS 에서 LHS 로 좌표계를 포팅하는 것은 두가지 주요 작업을 포함한다.
첫째는 polygon 을 구성하는 index buffer 의 내용이 CCW 에서 CW 로
재정렬되어야 한다. 둘째는 오브젝트 pos 의 z 값과 camera z axis 방향이
반전되어야 한다. 위의 첫번째 그림은 RHS 의 상황이고 두번째 그림은
별다른 작업없이 좌표계를 LHS로 전환했을때 벌어지는 현상이다. 거울에 비처진 것처럼
반대로 화면에 그려졌다. 세번째 그림은 포팅작업을 통해 RHS에서의 화면과
LHS에서의 화면이 같다.

앞서 언급한 포팅 작업중 첫째 작업은 필요 없을 수 있다.
DirectX 의 경우 기본 컬링 모드는 D3DCULL_CCW 이다. 이것을
D3DCULL_CW 로 바꾸면 재정렬 작업은 필요 없게된다.

# Vertex Processing

확대축소(scaling), 회전(rotation)은 선형변환(linear transformation) 이다.
선형변환(linear transformation) 에 이동(translation) 까지 포함되면
affine transformation 이다.

![](img/transform.png)

object(local) space coordinates 를 world space coordinates
로 변환하는 것을 world transform 이라고 한다.
world space coordinates 를 camera space coordinates 로 변환하는 것을
view transform 이라고 한다.
view space coordinates 를 clip space coordinates 로 변환하는 것을
투영변환(projection transform) 이라고 한다. 
clip space coordinates 는 normalized device coordinates 로 변환된다.
normalized device coordinates 를 window space coordinates 로 변환하는
것을 viewport transform 이라고 한다. clip space coordinates 부터 시작되는
변환은 rasterization 단계에서 이루어지는 것일까? viewport transform은
rasterization 단계에서 실행되는 것은 확실하다.

![](img/projection_transform.png)

projection transform 은 좌측의 view fustumn 을 우측의 canonical view
volume 으로 찌그러트리는 것이다. canonical view volume 은 directx 의 경우
직육면체 형태(2, 2, 1) 이기 때문에 near plane 의 object 들은 상대적으로
크기가 커질 것이고 far plane 의 object 들은 상대적으로 크기가 작아질
것이다. opengl 의 경우 정육면체 형태(2, 2, 2)이다.

![](img/RHS_to_LHS_on_rasterization.png)

rasterization 단계는 LHS 를 사용한다. 이전 단계에서 오른손좌표계(RHS) 를
사용했다면 z좌표를 반전시켜야 한다.

```
world-space point = model matrix * model point
view-space point = view matrix * world point
w-divide for NDC <= clip coords = projection matrix * view point
window coords = windows(screen) matrix * NDC
```

[이것](http://www.realtimerendering.com/udacity/transforms.html)은 object space
cartesian coordinate 가 (4,5,3)인 큐브를 예로 three.js와 함께 설명 했다.
src는 [이곳](https://github.com/erich666/cs291/blob/master/demo/unit7-view-pipeline.js)을 참고하자.

![](img/normal_transform.png)

normal vector 를 변환하는 것은 vertex 를 변환하는 것과 다르게 처리되어야
한다. vertex 를 변환 할 때와 똑같은 방법으로 변환행렬 M 과 surface
normal vector 를 곱하면 변환후 표면에 수직이 되지 못한다.  `M` 대신
`(M^{-1})^{T}` 를 곱해야 한다. 다음은 surface normal n 과 변환행렬
`(M^{-1})^{T}` 을 곱한 것과 `(r^{'}-p^{'})` 이 수직임을 보여준다.
`((r^{'}-p^{'})^{T}` 는 행렬 곱셉을 위해 transpose 한 것이다.

![](img/normal_transform_eq.png)

# Rasterization

Rasterization 은 hard wired하다. 클리핑(clipping), 원근
나눗셈(perspective division), 뒷면 제거(back-face culling), 뷰포트
변환(view-port transform), 스캔 변환(scan conversion), z-culling 등의
요소로 구성된다.

클리핑(clipping) 은 canonical view volume 의 바깥쪽에 포함된 폴리곤을
잘라내는 과정이다.

원근 나눗셈(perspective division) 은 지금까지 사용했던
동차좌표계(homogenious coordinates system) 를 데카르트좌표계(cartesian
coordinate system) 으로 변환하는 과정이다. 예를 들어 동차좌표가 (x, y,
z, w) 라면 데카르트좌표는 (x/w, y/w, z/w) 가 된다.

![](img/viewport_transform.png)

normalized device coordinates 를 screen space coordinates 로 변환하는
것을 viewport transform 이라고 한다. screen space 는 RHS 를 이용한다.
canonical view volume 의 z 값은 추후 z-buffering 을 위해 사용된다.

![](img/scan_conversion_1.png)
![](img/scan_conversion_2.png)

viewport transform 후에 각각의 polygon 은 screen space 에서 내부가 특정한
색깔로 채워져 보이게 된다. 이때 채워지는 색깔은 fragment 라는 것의
color 속성을 읽어온 것이다. polygon 의 내부를 채우는 pixel 수 만큼
fragment 들이 존재한다.  이와 같은 fragment 들을 생성하는 것을
스캔변환(scan conversion) 이라고 한다.  polygon 을 구성하는 vertex 3개를
보간(interpolation) 해서 fragment 들을 생성한다.  fragment 는 pixel 에
해당하는 normal, texture coordinates, color, depth 등을 가지고 있다.

output merging 단계에서 z-buffering 을 이용해서 깊이검사를 하는 것보다 
rasterization 단계에서 z-culling 을 한다면 훨 씬 효율적이다.
z-culling 을 최대한 활용하고자 하는 목적으로 이른바 pre-z pass algorithm 이
제안되었다.

# Fragment Processing

lighting 및 texturing 을 수행한다.

# Output Merging

z-buffering, alpha blending 을 수행한다.

# Lighting

[lighting](/lighting/README.md)

# Shader Models

opengl 과 direct3D 는 shader 용어들이 다르다.

| opengl | directx |
|:-------|:--------|
|Vertex Shader|Vertex Shader|
|Tessellation Control Shader|Hull Shader|
|Tessellation Evaluation Shader|Domain Shader|
|Geometry Shader|Geometry Shader|
|Fragment Shader|Pixel Shader|
|Compute Shader|Compute Shader|

shader model 은 directX 에서 사용하는 shader 버저닝 방법이다.
opengl 은 어떻게 하지???

shader model 4 는 geometry shader, stream output 이 추가되었다. shader model 5 는 hull shader, tessellator, domain shader 가 추가되었다.

# Shader Language

* [cg](/cg/README.md)
* [hlsl](/hlsl/README.md)
* [glsl](/glsl/README.md)
* [unity3dshaderlab](/unity3dshaderlab/README.md)

# Curve

Bezier Curve

* [A Primer on Bézier Curves](https://pomax.github.io/bezierinfo/?fbclid=IwAR0R700bRj8zWooU6hb8VScq10ytZxGNwlyMLU59MPnfiY-TEx7FvaGkIiA)

Hermite Curve

Catmull-Rom Spline

# Bump Mapping

* [Bumpmapping](/bumpmapping/README.md)

# Environment Mapping

# Light Mapping

# Shadow Mapping

* [Shadow Mapping @ TIL](/unity3dshaderlab/README.md#shadow-mapping)

# Ambient Occlusion Mapping

# Deferred Shading

* geometry pass, lighting pass 와 같이 2 단계로 렌더링 한다. geometry pass 에서는 gbuffer 에 positions, normals, albedos, speculars 등을 저장한다. lighting pass 에서는 gbuffer 를 입력으로 이용하여 light 를 렌더링한다.
* forward shading 은 light 마다 fragment shader 를 실행하지만 deferred shading 은 fragment shader 를 한번 실행해서 여러개의 light 를 처리한다. light 의 개수가 많을 때는 deferred shading 이 효율적이다. 
* deferred shading 을 사용하기 위해서는 GPU 가 MRT 를 지원해야 한다.
* [8.1.deferred_shading @ github](https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/5.advanced_lighting/8.1.deferred_shading/deferred_shading.cpp) 은 deffered shading 을 opengl 로 구현했다.

# Animation

# Collision Detection