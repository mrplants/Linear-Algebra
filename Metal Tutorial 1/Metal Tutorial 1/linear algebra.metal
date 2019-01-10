//
//  linear algebra.metal
//  Metal Tutorial 1
//
//  Created by Sean Fitzgerald on 1/8/19.
//  Copyright © 2019 Sean Fitzgerald. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

/*
 hadamardProduct:
 Perform an element-wise multiplication (hadamard product) of the two input matrices A and B, store the result in C
 */
kernel void hadamardProductKernel(texture_buffer<float, access::read> A [[texture(0)]],
																	texture_buffer<float, access::read> B [[texture(1)]],
																	texture_buffer<float, access::write> C [[texture(2)]],
																	uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) * B.read(gid), gid);
}
