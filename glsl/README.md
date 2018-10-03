<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Material](#material)
- [Basics](#basics)
  - [Drawing Red One](#drawing-red-one)
  - [Passing Data](#passing-data)
  - [Interface Blocks](#interface-blocks)
  - [Standard library](#standard-library)
  - [Variable types](#variable-types)
  - [Type qualifiers](#type-qualifiers)
  - [Interface blocks](#interface-blocks)
  - [Predefined variables](#predefined-variables)
- [Tutorial](#tutorial)
- [References](#references)

<!-- markdown-toc end -->

-------------------------------------------------------------------------------

# Abstract

glsl에 대해 정리한다.

# Material

* [The OpenGL ES Shading Language](https://www.khronos.org/files/opengles_shading_language.pdf)
  * glsl 1.0 manual
* [webgl 1.0 api quick reference card](https://www.khronos.org/files/webgl/webgl-reference-card-1_0.pdf)
  * glsl 포함 킹왕짱 요약 카드
* [openGL 4.5 API Reference Card](https://www.khronos.org/files/opengl45-quick-reference-card.pdf) 
  * glsl 포함 킹왕짱 요약 카드
* [webgl workshop](https://github.com/stackgl/webgl-workshop)
  * webgl 연습문제
* [shader school](https://github.com/stackgl/shader-school)
  * glsl 연습문제
* [fragment foundry](http://hughsk.io/fragment-foundry/chapters/01-hello-world.html)
  * fragment shader 연습문제
* [opengl shading language wiki](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)
* [WebGL Shaders and GLSL @ webglfundamentals](https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html)
  * glsl은 간편하게 webgl에서 학습하자.
* [Interactive 3D Graphics @ udacity](https://www.udacity.com/course/interactive-3d-graphics--cs291)
  * webgl을 이용한 3d graphics의 이론
* [Physically-Based Rendering in glTF 2.0 using WebGL](https://github.com/KhronosGroup/glTF-WebGL-PBR)
  * [PBR](/pbr/) on WebGL

# Basics

## Drawing Red One

화면 우측 중앙에 빨간 점 하나를 그리자.

```c
#version 450 core
void main(void)
{
  gl_Position = vec4(0.0, 0.0, 0.5, 1.0);
}
```

```c
#version 450 core out vec4 color; void main(void)
{
  color = vec4(0.0, 0.8, 1.0, 1.0); 
}
```

## Passing Data

```cpp
// Our rendering function
virtual void render(double currentTime)
{
  const GLfloat color[] = { (float)sin(currentTime) * 0.5f + 0.5f,
     
  (float)cos(currentTime) * 0.5f + 0.5f, 0.0f, 1.0f };
  glClearBufferfv(GL_COLOR, 0, color);
  // Use the program object we created earlier for rendering
  glUseProgram(rendering_program);
  GLfloat attrib[] = { (float)sin(currentTime) * 0.5f, (float)cos(currentTime) * 0.6f, 0.0f, 0.0f };
  // Update the value of input attribute 0
  glVertexAttrib4fv(0, attrib);
  // Draw one triangle
  glDrawArrays(GL_TRIANGLES, 0, 3);
}
```

```c
#version 450 core
// 'offset' and 'color' are input vertex attributes 
layout (location = 0) in vec4 offset;
layout (location = 1) in vec4 color;
// 'vs_color' is an output that will be sent to the next shader stage 
out vec4 vs_color;
void main(void)
{
  const vec4 vertices[3] = vec4[3](
    vec4(0.25, -0.25, 0.5, 1.0), 
    vec4(-0.25, -0.25, 0.5, 1.0), 
    vec4(0.25, 0.25, 0.5, 1.0));
// Add 'offset' to our hard-coded vertex position 
  gl_Position = vertices[gl_VertexID] + offset;
  // Output a fixed value for vs_color
  vs_color = color;
}
```

```c
#version 450 core
// Input from the vertex shader
in vec4 vs_color;
// Output to the framebuffer
out vec4 color;
void main(void) {
  // Simply assign the color we were given by the vertex shader to our output
  color = vs_color;
}
```

## Interface Blocks

`in, out` 변수들이 두개 이상인 경우 블록으로 만들어 사용하면 편하다.

```c
#version 450 core
// 'offset' is an input vertex attribute
layout (location = 0) in vec4 offset; 
layout (location = 1) in vec4 color;
// Declare VS_OUT as an output interface block 
// Send color to the next stage
out VS_OUT {
} vs_out;
vec4 color;

void main(void)
{
  const vec4 vertices[3] = vec4[3](
    vec4(0.25, -0.25, 0.5, 1.0), 
    vec4(-0.25, -0.25, 0.5, 1.0), 
    vec4(0.25, 0.25, 0.5, 1.0));
  // Add 'offset' to our hard-coded vertex position 
  gl_Position = vertices[gl_VertexID] + offset;
  // Output a fixed value for vs_color
  vs_out.color = color;
}
```

```c
#version 450 core
// Declare VS_OUT as an input interface block 
in VS_OUT {
  vec4 color; // Send color to the next stage
} fs_in;

// Output to the framebuffer
out vec4 color;
void main(void) {
  // Simply assign the color we were given by the vertex shader to our  output
  color = fs_in.color;
}
```

## Standard library

## Variable types

* uniform

application에서 값을 전달 받고 rendering pipeline에서 변하지 않는
값이다. vertex, fragment shader에서 사용할 수 있고 읽기전용이다.

* attribute

application에서 값을 전달 받고 rendering pipeline에서 변하지 않는
값이다. vertex shader에서 사용할 수 있고 읽기전용이다. vertex별로
값이 다를 수 있다.

* varying

vertex shader에서 fragment shader로 값을 전달 하는 경우 사용한다.
vertex shader에서 읽기 쓰기 가능하다. fragment shader에서는
interpolate된 값이 넘어온다.

## Type qualifiers

## Interface blocks

## Predefined variables

# Tutorial

* simple twgl
  * [demo](https://rawgit.com/iamslash/TIL/master/glsl/ex/a.html)
  * [src](ex/a.html)

# References

* [shader editor](http://shdr.bkcore.com/)
  * 웹에서 glsl shader를 실행해 볼 수 있다.
