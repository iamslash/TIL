# intro

- shader를 이해하기 전에 rendering pipeline을 이해해야한다. [1.2.3 The Graphics Hardware Pipeline](http://http.developer.nvidia.com/CgTutorial/cg_tutorial_chapter01.html)을
  참고하자. 꼭 읽어본다.
  - pixel과 fragment의 차이는 뭘까???
  - culling은 Primitive Assembly and Rasterization단계에서 수행된다.
  - depth testing, blengding, stencil-based shadowing은 Raster Operatios단계에서 수행된다.
- unity3d는 shader lab이라는 language로 shader를 표현한다.
- unity3d shader lab은 fixed function과 programmable pipeline으로 표현할 수 있다.
- programmable pipeline에는 vertex, fragment, surface shader가 있다.
- shader lab은 여러개의 subshader로 구성되어 있다. subshader는
  여러개의 pass로 구성될 수 있다. subshader는 하드웨어의 성능이 열악한 순서대로 기록한다.
- shader lab은 중간에 cg를 사용할 것을 추천한다.  
- vertex shader는 vertex를 기준으로 연산한다. fragment shader는
  pixel을 기준으로 연산한다. fragment shader가 vertext shader보다 더
  많이 호출된다.
  - [cg tutorial](http://http.developer.nvidia.com/CgTutorial/cg_tutorial_chapter10.html)에
  다음과 같은 언급이 있다.  A fragment program executes for each
  fragment that is generated, so fragment programs typically run
  several million times per frame. On the other hand, vertex programs
  normally run only tens of thousands of times per frame
- suface shader로 작성하면 vertex, fragment shader로 코드가 변환되고 컴파일된다.
- fixed function shader로 작성하면 내부적으로 shader import time에
  vertex, fragment shader로 변환된다.

# The Graphics Hardware Pipeline

![The Graphics Hardware Pipeline](http://http.developer.nvidia.com/CgTutorial/elementLinks/fig1_3.jpg)
![Types of Geometric Primitives](http://http.developer.nvidia.com/CgTutorial/elementLinks/fig1_4.jpg)
![Standard OpenGL and Direct3D Raster Operations](http://http.developer.nvidia.com/CgTutorial/elementLinks/fig1_5.jpg)

# tutorial

- [fixed function shader tutorial](https://docs.unity3d.com/Manual/ShaderTut1.html)
- [vertex, fragment shader tutorial](https://docs.unity3d.com/Manual/SL-VertexFragmentShaderExamples.html)
- [surface shader tutorial](https://docs.unity3d.com/Manual/SL-SurfaceShaderExamples.html)

# usage

- 빨간 색으로 칠하자.

```
Shader "Custom/SolidShader" {
    SubShader { 
        Pass {
            Color (1,0,0,1)
        } 
    } 
}
```

- 왜곡 효과를 주자.

```
```


# reference

- [Unity3d Shader Reference](https://docs.unity3d.com/Manual/SL-Reference.html)
- [nvidia cg tutorial](http://http.developer.nvidia.com/CgTutorial/cg_tutorial_chapter01.html)
- [Resources for Writing Shaders in Unity](https://github.com/VoxelBoy/Resources-for-Writing-Shaders-in-Unity)
- [a gentle introduction to shaders in unity3d](http://www.alanzucconi.com/2015/06/10/a-gentle-introduction-to-shaders-in-unity3d/)
- [Unity 5.x Shaders and Effects Cookbook](https://books.google.co.kr/books?id=-llLDAAAQBAJ&printsec=frontcover&dq=unity3d+5.x+shader+cook+book&hl=ko&sa=X&redir_esc=y#v=onepage&q=unity3d%205.x%20shader%20cook%20book&f=false)
