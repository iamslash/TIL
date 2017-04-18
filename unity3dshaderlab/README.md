# intro

- unity3d는 shader lab이라는 language로 shader를 표현한다.
- unity3d shader lab은 fixed function과 programmable pipeline으로 표현할 수 있다.
- programmable pipeline에는 vertex, fragment, surface shader가 있다.
- shader lab은 여러개의 subshader로 구성되어 있다. subshader는
  여러개의 pass로 구성될 수 있다. subshader는 하드웨어의 성능이 열악한 순서대로 기록한다.
- shader lab은 중간에 cg를 사용할 것을 추천한다.
- suface shader로 작성하면 vertex, fragment shader로 코드가 변환되고 컴파일된다.
- fixed function shader로 작성하면 내부적으로 shader import time에
  vertex, fragment shader로 변환된다.

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
