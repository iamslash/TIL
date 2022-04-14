# Abstract

2D Character Set Up 에 대해 정리한다.

# Materials

* [How to Make a 2D Character Creator in Unity | Youtube](https://www.youtube.com/watch?v=PNWK5o9l54w)
  * [2D-Character-Creator | github](https://github.com/tutmo/2D-Character-Creator)
  * [AnimatorOverrideController | unity](https://docs.unity3d.com/ScriptReference/AnimatorOverrideController.html)

# Summary

1. Create animations for each body part.
2. Make a parent game object with an Animator attached.
3. For each body part, make a child game object with a Sprite Renderer attached.
4. Add layers in the animator for each body part.
5. Use Scriptable Objects to store animations for each body part.
6. Write a script to select body parts.
7. Write a script to update body parts using an animator override controller.
