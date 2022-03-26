#!/bin/bash

rm -rf shader.vert.spv
rm -rf shader.frag.spv

glslangValidator -V shader.vert
glslangValidator -V shader.frag

mv vert.spv shader.vert.spv
mv frag.spv shader.frag.spv
