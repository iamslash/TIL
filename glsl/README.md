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
    - [color with normal value](#color-with-normal-value)
    - [bar](#bar)
- [References](#references)

<!-- markdown-toc end -->

-------------------------------------------------------------------------------

# Abstract

glsl에 대해 정리한다.

# Material

* [opengl shading language wiki](https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language)

# Basic

## Standard library

## Variable types

* uniform

application에서 값을 전달 받고 rendering pipeline에서 변하지 않는 값이다.

* attribute

* varying

## Type qualifiers

## Interface blocks

## Predefined variables

# Tutorial

## color with normal value

```glsl
precision highp float;
attribute vec3 position;
attribute vec3 normal;
uniform mat3 normalMatrix;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
varying vec3 fNormal;
varying vec3 fPosition;

void main()
{
  fNormal = normalize(normalMatrix * normal);
  vec4 pos = modelViewMatrix * vec4(position, 1.0);
  fPosition = pos.xyz;
  gl_Position = projectionMatrix * pos;
}
```

```glsl
precision highp float;
uniform float time;
uniform vec2 resolution;
varying vec3 fPosition;
varying vec3 fNormal;

void main()
{
  gl_FragColor = vec4(fNormal, 1.0);
}
```

## bar

# References

* [shader editor](http://shdr.bkcore.com/)
  * 웹에서 glsl shader를 실행해 볼 수 있다.
