# intro

- shader를 이해하기 전에 rendering pipeline을 이해해야한다. [1.2.3 The Graphics Hardware Pipeline](http://http.developer.nvidia.com/CgTutorial/cg_tutorial_chapter01.html)을
  참고하자. 꼭 읽어본다.
- unity3d는 shader lab이라는 language로 shader를 표현한다.
- unity3d shader lab은 fixed function과 programmable pipeline으로 표현할 수 있다.
- programmable pipeline에는 vertex, fragment, surface shader가 있다.
- shader lab은 여러개의 subshader로 구성되어 있다. subshader는
  여러개의 pass로 구성될 수 있다. subshader는 하드웨어의 성능이 열악한 순서대로 기록한다.
- shader lab은 중간에 cg를 사용할 것을 추천한다.
  - cg는 nvidia가 microsoft와 함께 개발한 shading
    language이다. directx, opengl을 지원한다. 그래서 unity3d shader
    lab이 cg사용을 권고하는 것 같다. 하지만 2012년 이후 개발이
    중단됬다. directx의 hlsl, opengl의 glsl은 사용할 일이 없을까???
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

- Vertex Transformation
  - vertex의 screen position을 만들어낸다.
  - vertex의 텍스처 좌표를 만들어낸다.
  - vertex의 라이팅 정보를 만들어낸다.
- Primitive Assembly and Rasterization
  - 이전 단계에서 전달된 vertex들은 geometric primitive들로 조립된다.
  - 조립된 primitive들은 view frustum clipping, back face culling된다. 
  - clipping, culling에서 살아남은 polygon들은 rasterize되어 fragment들을 만들어낸다.
  - fragment는 번데기 pixel은 나비와 같다. fragment는 여러 처리를
    거쳐서 framebuffer의 pixel로 전환된다. 그래서 potential
    pixel이라고 한다.
    - A fragment has an associated pixel location, a depth value, and a
    set of interpolated parameters such as a color, a secondary
    (specular) color, and one or more texture coordinate sets.
- Interpolation, Texturing, and Coloring
  - fragment의 parameter들을 interpolate하고 fragment의 final color를 결정한다.
- Raster Operations
  - fragment마다 실행된다. depth testing, blending, sencil test가 수행된다.
  - 결국 frame buffer는 최종 처리된 컬러값을 쓴다.

![The Graphics Hardware Pipeline](http://http.developer.nvidia.com/CgTutorial/elementLinks/fig1_3.jpg)
![Types of Geometric Primitives](http://http.developer.nvidia.com/CgTutorial/elementLinks/fig1_4.jpg)
![Standard OpenGL and Direct3D Raster Operations](http://http.developer.nvidia.com/CgTutorial/elementLinks/fig1_5.jpg)
![Visualizing the Graphics Pipeline](http://http.developer.nvidia.com/CgTutorial/elementLinks/fig1_6.jpg)
![The Programmable Graphics Pipeline](http://http.developer.nvidia.com/CgTutorial/elementLinks/fig1_7.jpg)

# tutorial reference

- [fixed function shader tutorial](https://docs.unity3d.com/Manual/ShaderTut1.html)
- [vertex, fragment shader tutorial](https://docs.unity3d.com/Manual/SL-VertexFragmentShaderExamples.html)
- [surface shader tutorial](https://docs.unity3d.com/Manual/SL-SurfaceShaderExamples.html)

# usage

- 빨간 색으로 칠하자.

```
Shader "Custom/A" {
    SubShader { 
        Pass {
            Color (1,0,0,1)
        } 
    } 
}
```

- Direct Light를 배치하고 Cube의 표면에 Diffuse, Ambient를 적용하자. 
  - Direct Light를 활성화 비활성화 해보면 차이를 알 수 있다.
  - Lighting On은 Material이 없으면 작동하지 않는다.

```
Shader "Custom/A"{ 
    SubShader { 
        Pass { 
            Material { 
                Diffuse (1,1,1,1) 
                Ambient (1,1,1,1) 
            } 
            Lighting On 
        } 
    } 
}
```

- Properties를 이용하여 Diffuse, Ambient를 조정할 수 있게 해보자.

```
Shader "Custom/A"{ 
    Properties { 
        _MyColor ("Main Color", COLOR) = (0,0,1,1) 
    } 
    SubShader { 
        Pass { 
            Material { 
                Diffuse [_MyColor] 
                Ambient [_MyColor] 
            } 
            Lighting On 
        } 
    } 
} 
```

- Properties를 이용하여 texture를 지정해 보자.

```
Shader "Custom/A" { 
    Properties { 
        _MyColor ("Main Color", COLOR) = (1,1,1,1) 
        _MainTex ("Base Texture", 2D) = "white" {} 
    } 
    SubShader { 
        Pass { 
            Material { 
                Diffuse [_MyColor] 
                Ambient [_MyColor] 
            } 
            Lighting On 

            SetTexture [_MainTex]  
        } 
    } 
} 
```

- texture는 color와 blending되고 light적용도 받게 해보자.

```
Shader "Custom/A"{ 
    Properties { 
        _MyColor ("Main Color", COLOR) = (0,0,1,1) 
        _MainTex ("Base Texture", 2D) = "white" {} 
    } 
    SubShader { 
        Pass { 
            Material { 
                Diffuse [_MyColor] 
                Ambient [_MyColor] 
            } 
            Lighting On 

            SetTexture [_MainTex] { 
                    Combine texture * primary DOUBLE 
            } 
        } 
    } 
} 
```

- 두장의 텍스처를 섞어 보자.
  - A lerp(B) C 의 의미는 다음과 같다.
    - B의 alpha값을 확인해서 A와 B값을 보간하여 사용한다.
    - B의 alpha값이 1이면 A를 사용하고 0이면 B를 사용한다.

```
Shader "Custom/A"{ 
    Properties { 
         _MainTex("Texture", 2D) = "white" {} 
         _SubTex("Texture", 2D) = "white" {} 
    } 
    SubShader { 
           Pass { 
               SetTexture [_MainTex] { 
                 Combine texture
               } 
                                                 
               SetTexture [_SubTex] { 
                 Combine texture lerp(texture) previous 
               } 
           } 
    } 
}
```

- 반 투명한 물체를 만들어 보자.
  - _Color의 alpha값을 조정해보니 투명해진다. 하지만 가려진 물체는 안그려진다. 왜지???
  

```
Shader "Custom/A"{ 
    Properties { 
        _Color ("Main Color", COLOR) = (1,1,1,1) 
        _MainTex("Texture", 2D) = "white" {} 
        _SubTex("Texture", 2D) = "white" {} 
    } 
    SubShader { 
        Pass { 
                 Blend SrcAlpha OneMinusSrcAlpha 
                
                 SetTexture [_MainTex] { 
                          Combine texture 
                 } 
                                                                       
                 SetTexture [_SubTex] { 
                          ConstantColor[_Color] 
                          Combine texture lerp(texture) previous, constant 
                 }
        } 
    } 
} 
```

- cg를 이용해서 simple unlit을 만들어 보자.
  - Light와 상관없이 텍스처를 보여주자.
  - : POSITION, : TEXCOORD0, : SV_Target 등은 semantic이라고 한다.
    - cg에게 해당 변수를 어떤 용도로 사용하겠다는 의도를 표현한다.
    - [이곳](https://docs.unity3d.com/Manual/SL-ShaderSemantics.html)에 자세한 설명이 있다.

```cg
Shader "Custom/A"
{
    Properties
    {
        // we have removed support for texture tiling/offset,
        // so make them not be displayed in material inspector
        [NoScaleOffset] _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Pass
        {
            CGPROGRAM
            // use "vert" function as the vertex shader
            #pragma vertex vert
            // use "frag" function as the pixel (fragment) shader
            #pragma fragment frag

            // vertex shader inputs
            struct appdata
            {
                float4 vertex : POSITION; // vertex position
                float2 uv : TEXCOORD0; // texture coordinate
            };

            // vertex shader outputs ("vertex to fragment")
            struct v2f
            {
                float2 uv : TEXCOORD0; // texture coordinate
                float4 vertex : SV_POSITION; // clip space position
            };

            // vertex shader
            v2f vert (appdata v)
            {
                v2f o;
                // transform position to clip space
                // (multiply with model*view*projection matrix)
                o.vertex = mul(UNITY_MATRIX_MVP, v.vertex);
                // just pass the texture coordinate
                o.uv = v.uv;
                return o;
            }
            
            // texture we will sample
            sampler2D _MainTex;

            // pixel shader; returns low precision ("fixed4" type)
            // color ("SV_Target" semantic)
            fixed4 frag (v2f i) : SV_Target
            {
                // sample texture and return it
                fixed4 col = tex2D(_MainTex, i.uv);
                return col;
            }
            ENDCG
        }
    }
}
```

- vertex, fragment shader를 이용하여 한가지 색으로 칠해보자.

```
// Upgrade NOTE: replaced 'mul(UNITY_MATRIX_MVP,*)' with 'UnityObjectToClipPos(*)'

Shader "Unlit/SingleColor"
{
    Properties
    {
        // Color property for material inspector, default to white
        _Color ("Main Color", Color) = (1,1,1,1)
    }
    SubShader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            // vertex shader
            // this time instead of using "appdata" struct, just spell inputs manually,
            // and instead of returning v2f struct, also just return a single output
            // float4 clip position
            float4 vert (float4 vertex : POSITION) : SV_POSITION
            {
                return UnityObjectToClipPos(vertex);
            }
            
            // color from the material
            fixed4 _Color;

            // pixel shader, no inputs needed
            fixed4 frag () : SV_Target
            {
                return _Color; // just return it
            }
            ENDCG
        }
    }
}

```

- normal맵을 적용해보자.

```
Shader "Unlit/WorldSpaceNormals"
{
    // no Properties block this time!
    SubShader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct v2f {
                half3 worldNormal : TEXCOORD0;
                float4 pos : SV_POSITION;
            };

            v2f vert (float4 vertex : POSITION, float3 normal : NORMAL)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(vertex);
                o.worldNormal = UnityObjectToWorldNormal(normal);
                return o;
            }
            
            fixed4 frag (v2f i) : SV_Target
            {
                fixed4 c = 0;
                c.rgb = i.worldNormal*0.5+0.5;
                return c;
            }
            ENDCG
        }
    }
}

```

- skymap을 reflect해보자.

```
Shader "Unlit/SkyReflection"
{
    SubShader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct v2f {
                half3 worldRefl : TEXCOORD0;
                float4 pos : SV_POSITION;
            };

            v2f vert (float4 vertex : POSITION, float3 normal : NORMAL)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(vertex);
                // compute world space position of the vertex
                float3 worldPos = mul(_Object2World, vertex).xyz;
                // compute world space view direction
                float3 worldViewDir = normalize(UnityWorldSpaceViewDir(worldPos));
                // world space normal
                float3 worldNormal = UnityObjectToWorldNormal(normal);
                // world space reflection vector
                o.worldRefl = reflect(-worldViewDir, worldNormal);
                return o;
            }
        
            fixed4 frag (v2f i) : SV_Target
            {
                // sample the default reflection cubemap, using the reflection vector
                half4 skyData = UNITY_SAMPLE_TEXCUBE(unity_SpecCube0, i.worldRefl);
                // decode cubemap data into actual color
                half3 skyColor = DecodeHDR (skyData, unity_SpecCube0_HDR);
                // output it!
                fixed4 c = 0;
                c.rgb = skyColor;
                return c;
            }
            ENDCG
        }
    }
}
```

# reference

- [Unity3d Shader Reference](https://docs.unity3d.com/Manual/SL-Reference.html)
- [nvidia cg tutorial](http://http.developer.nvidia.com/CgTutorial/cg_tutorial_chapter01.html)
- [Resources for Writing Shaders in Unity](https://github.com/VoxelBoy/Resources-for-Writing-Shaders-in-Unity)
- [a gentle introduction to shaders in unity3d](http://www.alanzucconi.com/2015/06/10/a-gentle-introduction-to-shaders-in-unity3d/)
- [Unity 5.x Shaders and Effects Cookbook](https://books.google.co.kr/books?id=-llLDAAAQBAJ&printsec=frontcover&dq=unity3d+5.x+shader+cook+book&hl=ko&sa=X&redir_esc=y#v=onepage&q=unity3d%205.x%20shader%20cook%20book&f=false)
