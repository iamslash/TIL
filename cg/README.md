# Abstract

cg는 NVIDIA와 MS가 함께 개발한 언어이다. 3.1 버전을 끝으로 더이상
개발되지 않는다. 그러나 unity shaderlab에서는 cg사용을 권고하고
있다. cg가 OpenGL과 DirectX를 동시에 지원하기 때문일 것이다.

# Materials

* [The Cg Tutorial @ nvidia](http://developer.download.nvidia.com/CgTutorial/cg_tutorial_chapter01.html)
  * 최고의 cg 책
* [cg Toolkit](https://developer.nvidia.com/cg-toolkit)
  * NVIDIA가 개발한 cg toolkit이다. 설치하면 다양한 예제를 확인할 수 있다.
* [nvidia shader library](http://developer.download.nvidia.com/shaderlibrary/webpages/shader_library.html)
  * HLSL, CgFX등 다양한 예제

# Basic


## CgFX

CgFX는 vertex, fragment shader의 wrapper이다. unity shaderlab과
유사하다. technique, pass, render state등으로 구성되어 있다.
render state은 vertex, fragment shader를 컴파일한다.

* sample cgFX file : calculates basic diffuse, specular lighting

```cg
struct VS_INPUT
{
  float4 vPosition  : POSITION;
  float4 vNormal    : NORMAL;
  float4 vTexCoords : TEXCOORD0;
};

struct VS_OUTPUT
{
  float4 vTexCoord0 : TEXCOORD0;
  float4 vDiffuse   : COLOR0;
  float4 vPosition  : POSITION;
  float4 vSpecular  : COLOR1;
};

VS_OUTPUT myvs(uniform 
               float4x4 ModelViewProj,
               uniform 
               float4x4 ModelView,
               uniform 
               float4x4 ModelViewIT,
               uniform 
               float4x4 ViewIT,
               uniform 
               float4x4 View,
               const VS_INPUT vin,
               uniform 
               float4 lightPos,
               uniform 
               float4 diffuse,
               uniform 
               float4 specular,
               uniform 
               float4 ambient)
{
  VS_OUTPUT vout;
  float4 position = mul(ModelView, vin.vPosition);
  float4 normal =   mul(ModelViewIT, vin.vNormal);
  float4 viewLightPos = mul(View, lightPos);
  float4 lightvec = normalize(viewLightPos - position);
  float4 eyevec =   normalize(ViewIT[3]);
  float self_shadow = max(dot(normal, lightvec), 0);
  float4 halfangle = normalize(lightvec + eyevec);
  float spec_term =  max(dot(normal, halfangle), 0);
  float4 diff_term = ambient + diffuse * self_shadow +
      self_shadow * spec_term * specular;
  vout.vDiffuse = diff_term;
  vout.vPosition = mul(ModelViewProj, vin.vPosition);
  return vout;
}

float4x4 vit     : ViewIT;
float4x4 viewmat : View;
float4x4 mv      : WorldView;
float4x4 mvit    : WorldViewIT;
float4x4 mvp     : WorldViewProjection;
float4 diffuse   : DIFFUSE = { 0.1f, 0.1f, 0.5f, 1.0f };
float4 specular  : SPECULAR = { 1.0f, 1.0f, 1.0f, 1.0f };
float4 ambient   : AMBIENT = { 0.1f, 0.1f, 0.1f, 1.0f };
float4 lightPos  : Position
<
string Object = "PointLight";
string Space = "World";
> = { 100.0f, 100.0f, 100.0f, 0.0f };
technique t0
{
  pass p0
  {
    Zenable = true;
    ZWriteEnable = true;
    CullMode = None;

    VertexShader = compile 
        vs_1_1 myvs( mvp, mv, mvit, vit,
                     viewmat, lightPos,
                     diffuse, specular,
                     ambient);
  }
}
```
