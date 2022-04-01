#!/bin/bash

rm -rf shaders/*.spv

for f in shaders/*.{vert,frag}; do
	echo "Compiling $f file...";
	glslangValidator -V $f -o $f.spv
done
