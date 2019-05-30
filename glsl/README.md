- [Abstract](#abstract)
- [Material](#material)
- [References](#references)
- [Basics](#basics)
  - [Version](#version)
  - [Data Types](#data-types)
  - [Function Declaration](#function-declaration)
  - [Variable Declaration](#variable-declaration)
  - [Type Constructors](#type-constructors)
  - [Array Declaration](#array-declaration)
  - [Array Accessors](#array-accessors)
  - [Swizzling](#swizzling)
  - [Comments](#comments)
  - [Function Calls](#function-calls)
  - [Control Structures](#control-structures)
  - [Operators](#operators)
  - [Builtin Functions](#builtin-functions)
  - [Red Dot](#red-dot)
  - [Passing Data](#passing-data)
  - [Interface Blocks](#interface-blocks)
  - [Variable Types](#variable-types)
  - [Built-in Variables](#built-in-variables)
- [Tutorial](#tutorial)

-------------------------------------------------------------------------------

# Abstract

glsl에 대해 정리한다.

# Material

* [shader playground](http://shader-playground.timjones.io/)
  * online compiler
  * [src](https://github.com/tgjones/shader-playground) 
- [OpenGL Programming Guide: The Official Guide to Learning OpenGL, Version 4.3](http://www.opengl-redbook.com/)
  - opengl red book
  - [src](https://github.com/openglredbook/examples)
- [OpenGL Superbible: Comprehensive Tutorial and Reference](http://www.openglsuperbible.com)
  - opengl blue book
  - [src](https://github.com/openglsuperbible/sb7code)
- [OpenGL Shading Language](https://www.amazon.com/OpenGL-Shading-Language-Randi-Rost/dp/0321637631/ref=sr_1_1?ie=UTF8&qid=1538565859&sr=8-1&keywords=opengl+shading+language)
  - opengl orange book 
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

# References

* [shader editor](http://shdr.bkcore.com/)
  * 웹에서 glsl shader를 실행해 볼 수 있다.

# Basics

## Version

shader 의 첫 줄에 다음과 같은 형식의 버전이 필요하다.

```c
#version 450 core
```

다음은 opengl 과 glsl 의 버전목록이다. 

| GLSL Version | OpenGL Version | Date | Shader Preprocessor |
|:--|:--|:--|:--|
| 1.10.59 |	2.0 |	30 April 2004 |	`#version 110` |
| 1.20.8  |	2.1	| 07 September 2006 |	`#version 120` |
| 1.30.10 |	3.0	| 22 November 2009	| `#version 130` |
| 1.40.08 |	3.1	| 22 November 2009	| `#version 140` |
| 1.50.11 |	3.2	| 04 December 2009	| `#version 150` |
| 3.30.6  |	3.3	| 11 March 2010 | `#version 330` |
| 4.00.9  |	4.0	| 24 July 2010	| `#version 400` |
| 4.10.6  |	4.1	| 24 July 2010	| `#version 410` |
| 4.20.11 |	4.2	| 12 December 2011	| `#version 420` |
| 4.30.8  |	4.3	| 7 February 2013	  | `#version 430` |
| 4.40.9  |	4.4	| 16 June 2014	    | `#version 440` |
| 4.50.7  | 4.5	| 09 May 2017	      | `#version 450` |
| 4.60.5  |	4.6 |	14 June 2018	    | `#version 460` |

## Data Types

주로 사용하는 데이터형은 다음과 같다.

```c
float      int
vec2  mat2 ivec2 
vec3  mat3 ivec3
vec4  mat4 ivec4

Sampler2D  bool
```

## Function Declaration

주요 함수 정의 방법은 다음과 같다.

```c
float computeSum(float a, float b) {
  return a + b;
}

float computeSum(in float a, in float b) {
  return a + b;
}

float computeSum(out float s, in float a, in float b) {
  s = a + b;
}
```

## Variable Declaration

주요 변수 선언 방법은 다음과 같다.

```c
void A() {
  float x = 1.0, y = 2.0;
}
```

## Type Constructors

```c
vec3 color = vec3(0,0, 0.5, 1.0);

mat3 im = mat3(
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0
);

vec3 color = vec3(1.0);

vec3 color = vec3(0.0, 0.5, 1.0);
vec4 colorAlpha = vec4(color, 1.0);
```

## Array Declaration

```c
vec2[3] coords = vec2[3](
  vec2(0.2, 0.3),
  vec2(0.8, 0.4),
  vec2(0.5, 0.7)
);
```

## Array Accessors

```c
void A(vec2[3] coords) {
  vec2 firstCoord = coords[0];
  float x = firstCoord[0];
}
```

## Swizzling

```c
void A(vec2 pos) {
  float x = pos.x;
  pos.y = 0.0;
}
// [0] .x .r .s
// [1] .y .g .t
// [2] .z .b .p
// [3] .w .a .q

void B(vec4 col) {
  vec3 c = col.rgb;
  vec3 cbgr = col.bgr;
  vec4 grayscale = col.rrra;
}
```

## Comments

```c
/* ...
  */
// ...
```

## Function Calls

```c
float y = A(x);
float z = B(0.5, 1.5, 0.0);
```

## Control Structures

```c
for (int i = 0; i < N; ++i) {
  
}

int i = 0;
while (i < N) {

}

if (a == b) {

} else {

}

return; return X;
break;
continue;
discard;
```

## Operators

주요 연산자의 우선순위는 다음과 같다.

```c
++ --
+ - ! ~
* / %
+ -
<< >>
< <= >= >
== !=
&
^
|
&&
||
? :
```

```c
//vector * matrix
//matrix * vector
//matrix * matrix

vec3(x1, y1, z1) * vec3(x2, y2, z2) == vec3(x1 * x2, y1 * y2, z1 * z2)
```

## Builtin Functions

주요 도움함수들은 다음과 같다. 그 외의 것은 [이곳](https://www.khronos.org/registry/OpenGL-Refpages/gl4/index.php)을 참고하자.

```c
abs sign floor ceil
fract(x)
min(a, b) max(a, b) mod(x, modulo) 
sqrt(x) pow(x, exponent) exp(x) log(x)
sin cos tan asin acos atan

clamp(x, min, max) == min(max(x, min), max)
step(threshold, x)
smoothstep(start, end, x)
mix(a, b, ratio) == (1.0 - ratio) * a + ratio * b)
mix(RED, YELLO, pos.x)
length(vec2(x, y))
distance(A, B) == length(B - A)
normalize(vector) == vector / length(vector)
texture(aTexture, texCoord)
```

## Red Dot

화면 우측 중앙에 빨간 점 하나를 그리자.

```glsl
#version 450 core
void main(void)
{
  gl_Position = vec4(0.0, 0.0, 0.5, 1.0);
}
```

```glsl
#version 450 core 
out vec4 color; 
void main(void)
{
  color = vec4(0.0, 0.8, 1.0, 1.0); 
}
```

## Passing Data

```glsl
// Our rendering function
virtual void render(double currentTime)
{
  const GLfloat color[] = { 
    (float)sin(currentTime) * 0.5f + 0.5f,
    (float)cos(currentTime) * 0.5f + 0.5f, 0.0f, 1.0f };
  glClearBufferfv(GL_COLOR, 0, color);
  
  // Use the program object we created earlier for rendering
  glUseProgram(rendering_program);
  GLfloat attrib[] = { 
    (float)sin(currentTime) * 0.5f, 
    (float)cos(currentTime) * 0.6f, 0.0f, 0.0f };
  
  // Update the value of input attribute 0
  glVertexAttrib4fv(0, attrib);
  
  // Draw one triangle
  glDrawArrays(GL_TRIANGLES, 0, 3);
}
```

```glsl
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

```glsl
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

```glsl
#version 450 core

// 'offset' is an input vertex attribute
layout (location = 0) in vec4 offset; 
layout (location = 1) in vec4 color;

// Declare VS_OUT as an output interface block 
// Send color to the next stage
out VS_OUT {
  vec4 color;
} vs_out;

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

```glsl
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

## Variable Types

* uniform

Application 에서 vertex, fragment shader 에게 바뀌지 않는 값을 전달할 때 사용한다.

> * Attribute variables communicate frequently changing values from the application to a vertex shader, uniform variables communicate infrequently changing values from the application to any shader, and varying variables communicate interpolated values from a vertex shader to a fragment shader.

* attribute
 
Application 에서 vetex shader 에게 자주 바뀌는 값을 전달할 때 사용한다.

* varying

vertex shader 에서 fragment shader 로 보간된 값을 전달할 때 사용한다.

## Built-in Variables

`gl_VertexID` 처럼 쉐이더의 종류 마다 미리 정의된 변수들이 존재한다. [참고](https://www.khronos.org/opengl/wiki/Built-in_Variable_(GLSL))

# Tutorial

* simple twgl
  * [demo](https://rawgit.com/iamslash/TIL/master/glsl/ex/a.html)
  * [src](ex/a.html)