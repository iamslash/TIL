<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-refresh-toc -->
**Table of Contents**

- [Abstract](#abstract)
- [Material](#material)
- [Basic](#basic)
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

* [shader school](https://github.com/stackgl/shader-school)
  * glsl 연습문제
* [fragment foundry](http://hughsk.io/fragment-foundry/chapters/01-hello-world.html)
  * fragment shader 연습문제
* [webgl workshop](https://github.com/stackgl/webgl-workshop)
  * webgl 연습문제
* [opengl shading language wiki](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)
* [WebGL Shaders and GLSL](https://webglfundamentals.org/webgl/lessons/webgl-shaders-and-glsl.html)
  * glsl은 간편하게 webgl에서 학습하자.
* [Interactive 3D Graphics @ udacity](https://www.udacity.com/course/interactive-3d-graphics--cs291)
  * webgl을 이용한 3d graphics의 이론
* [Physically-Based Rendering in glTF 2.0 using WebGL](https://github.com/KhronosGroup/glTF-WebGL-PBR)
  * [PBR](/pbr/) on WebGL

# Basic

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

* [simple twgl](https://rawgit.com/iamslash/TIL/master/glsl/ex/a.html)
  * [src](ex/a.html)

# References

* [shader editor](http://shdr.bkcore.com/)
  * 웹에서 glsl shader를 실행해 볼 수 있다.
