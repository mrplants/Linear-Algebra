//
//  linear algebra.metal
//  Metal Tutorial 1
//
//  Created by Sean Fitzgerald on 1/8/19.
//  Copyright © 2019 Sean Fitzgerald. All rights reserved.
//

#include <metal_stdlib>
#include <metal_atomic>
#include "ShaderHelper.h"
using namespace metal;

#pragma ELEMENT_WISE_OPERATIONS
/////////////////////////////
// ELEMENT-WISE OPERATIONS //
/////////////////////////////

/*
 hadamardProductKernel:
 Perform an element-wise multiplication (hadamard product) of the two input matrices A and B, store the result in C
 */
kernel void hadamardProductKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
																	texture_buffer<float, access::read> B [[texture(MatrixRightArgument)]],
																	texture_buffer<float, access::write> C [[texture(MatrixResult)]],
																	uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) * B.read(gid), gid);
}

/*
 subtractKernel:
 Perform an element-wise subtraction of the two input matrices A and B, store the result in C
 */
kernel void subtractKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
													 texture_buffer<float, access::read> B [[texture(MatrixRightArgument)]],
													 texture_buffer<float, access::write> C [[texture(MatrixResult)]],
													 uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) - B.read(gid), gid);
}

/*
 addKernel:
 Perform an element-wise addition of the two input matrices A and B, store the result in C
 */
kernel void addKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
											texture_buffer<float, access::read> B [[texture(MatrixRightArgument)]],
											texture_buffer<float, access::write> C [[texture(MatrixResult)]],
											uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) + B.read(gid), gid);
}

/*
 multiplyStaticKernel:
 Perform an element-wise multiplication between the argument A matrix and a scalar b, store the result in C
 A[i,j] * b = C[i,j]
 */
kernel void multiplyScalarKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
																 const device float& b [[buffer(MatrixRightArgument)]],
																 texture_buffer<float, access::write> C [[texture(MatrixResult)]],
																 uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) * b,gid);
}

/*
 addStaticKernel:
 Perform an element-wise addition between the argument A matrix and a scalar b, store the result in C
 A[i,j] + b = C[i,j]
 */
kernel void addScalarKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
														const device float& b [[buffer(MatrixRightArgument)]],
														texture_buffer<float, access::write> C [[texture(MatrixResult)]],
														uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) + b,gid);
}

/*
 subtractStaticKernel:
 Perform an element-wise multiplication between the argument A matrix and a scalar b, store the result in C
 This can be used with a negative b if it is the actually the left argument
 A[i,j] - b = C[i,j]
 */
kernel void subtractScalarKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
																 const device float& b [[buffer(ScalarRightArgument)]],
																 texture_buffer<float, access::write> C [[texture(MatrixResult)]],
																 uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) - b,gid);
}

/*
 hadamardProductRowVectorKernel:
 Perform a row-by-row element-wise multiplication between the argument A matrix and a vector row B, store the result in C
 A[i,] * B = C[i,]
 */
kernel void hadamardProductRowVectorKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
																					 texture_buffer<float, access::read> B [[texture(MatrixRightArgument)]],
																					 texture_buffer<float, access::write> C [[texture(MatrixResult)]],
																					 const device int& numColumns [[buffer(ScalarWidth)]],
																					 uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) * B.read(gid % numColumns),gid);
}

/*
 addRowVectorKernel:
 Perform a row-by-row element-wise multiplication between the argument A matrix and a vector row B, store the result in C
 A[i,] + B = C[i,]
 */
kernel void addRowVectorKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
															 texture_buffer<float, access::read> B [[texture(MatrixRightArgument)]],
															 texture_buffer<float, access::write> C [[texture(MatrixResult)]],
															 const device int& numColumns [[buffer(ScalarWidth)]],
															 uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) + B.read(gid % numColumns),gid);
}

/*
 subtractRowVectorKernel:
 Perform a row-by-row element-wise multiplication between the argument A matrix and a vector row B, store the result in C
 This can be used with a negative B if it is the actually the left argument
 A[i,] - B = C[i,]
 */
kernel void subtractRowVectorKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
																		texture_buffer<float, access::read> B [[texture(MatrixRightArgument)]],
																		texture_buffer<float, access::write> C [[texture(MatrixResult)]],
																		const device int& numColumns [[buffer(ScalarWidth)]],
																		uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) - B.read(gid % numColumns),gid);
}

/*
 hadamardProductColumnVectorKernel:
 Perform a column-by-colomn element-wise multiplication between the argument A matrix and a vector row B, store the result in C
 A[i,] * B = C[i,]
 */
kernel void hadamardProductColumnVectorKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
																							texture_buffer<float, access::read> B [[texture(MatrixRightArgument)]],
																							texture_buffer<float, access::write> C [[texture(MatrixResult)]],
																							const device int& numColumns [[buffer(ScalarWidth)]],
																							uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) * B.read(gid / numColumns),gid);
}

/*
 addColumnVectorKernel:
 Perform a row-by-row element-wise multiplication between the argument A matrix and a vector row B, store the result in C
 A[i,] + B = C[i,]
 */
kernel void addColumnVectorKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
																	texture_buffer<float, access::read> B [[texture(MatrixRightArgument)]],
																	texture_buffer<float, access::write> C [[texture(MatrixResult)]],
																	const device int& numColumns [[buffer(ScalarWidth)]],
																	uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) + B.read(gid / numColumns),gid);
}

/*
 subtractColumnVectorKernel:
 Perform a row-by-row element-wise multiplication between the argument A matrix and a vector row B, store the result in C
 This can be used with a negative B if it is the actually the left argument
 A[i,] - B = C[i,]
 */
kernel void subtractColumnVectorKernel(texture_buffer<float, access::read> A [[texture(MatrixLeftArgument)]],
																			 texture_buffer<float, access::read> B [[texture(MatrixRightArgument)]],
																			 texture_buffer<float, access::write> C [[texture(MatrixResult)]],
																			 const device int& numColumns [[buffer(ScalarWidth)]],
																			 uint gid [[thread_position_in_grid]]) {
	C.write(A.read(gid) - B.read(gid / numColumns),gid);
}

#pragma COMMON_OPERATIONS
///////////////////////
// COMMON OPERATIONS //
///////////////////////

/*
 exponentKernel:
 Perform an element-wise operation, e^a[i,j] on the input matrix A. Store the result in C
 */
kernel void exponentKernel(texture_buffer<float, access::read> A [[texture(MatrixArgument1)]],
													 texture_buffer<float, access::write> C [[texture(MatrixResult)]],
													 uint gid [[thread_position_in_grid]]) {
	C.write(exp(A.read(gid)), gid);
}
