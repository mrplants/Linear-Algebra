//
//  Matrix.swift
//  Metal Tutorial 1
//
//  Created by Sean Fitzgerald on 1/8/19.
//  Copyright © 2019 Sean Fitzgerald. All rights reserved.
//

import Foundation
import Metal
// TODO: Include ShaderHelper.h using an objective-C bridging header
// FUTURE TODO: allow arbitrary number of axes

struct Matrix {
	// Static Properties
	static let GPU = MTLCreateSystemDefaultDevice()
	static let CommandQueue = Matrix.GPU?.makeCommandQueue()
	static let DefaultLibrary = Matrix.GPU?.makeDefaultLibrary()
	// Matrix-Matrix Kernels
	static let HadamardKernel = Matrix.DefaultLibrary?.makeFunction(name: "hadamardProductKernel")
	static let SubtractKernel = Matrix.DefaultLibrary?.makeFunction(name: "subtractKernel")
	static let AddKernel = Matrix.DefaultLibrary?.makeFunction(name: "addKernel")
	// Matrix-RowVector Kernels
	static let HadamardProductRowVectorKernel = Matrix.DefaultLibrary?.makeFunction(name: "hadamardProductRowVectorKernel")
	static let AddRowVectorKernel = Matrix.DefaultLibrary?.makeFunction(name: "addRowVectorKernel")
	static let SubtractRowVectorKernel = Matrix.DefaultLibrary?.makeFunction(name: "subtractRowVectorKernel")
	// Matrix-ColumnVector Kernels
	static let HadamardProductColumnVectorKernel = Matrix.DefaultLibrary?.makeFunction(name: "hadamardProductColumnVectorKernel")
	static let AddColumnVectorKernel = Matrix.DefaultLibrary?.makeFunction(name: "addColumnVectorKernel")
	static let SubtractColumnVectorKernel = Matrix.DefaultLibrary?.makeFunction(name: "subtractColumnVectorKernel")
	// Matrix-Scalar Kernels
	static let MultiplyScalarKernel = Matrix.DefaultLibrary?.makeFunction(name: "multiplyScalarKernel")
	static let AddScalarKernel = Matrix.DefaultLibrary?.makeFunction(name: "addScalarKernel")
	static let SubtractScalarKernel = Matrix.DefaultLibrary?.makeFunction(name: "subtractScalarKernel")
	// Common Parallel Operation Kernels
	static let ExponentKernel = Matrix.DefaultLibrary?.makeFunction(name: "exponentKernel")

	let dataTexture:MTLTexture
	let width:Int
	let height:Int
	
	init(data:[Float], width:Int, height:Int) {
		// TODO: Build an init that accepts an array with a structure
		self.width = width
		self.height = height
		let texDescriptor = MTLTextureDescriptor.textureBufferDescriptor(with: .r32Float,
																																		 width: self.width * self.height,
																																		 resourceOptions: .storageModeManaged,
																																		 usage: [.shaderRead, .shaderWrite])
		guard
			let dataTexture = Matrix.GPU?.makeTexture(descriptor: texDescriptor)
			else {
				print("Could not create texture in Matrix initializer")
				exit(EXIT_FAILURE)
		}
		self.dataTexture = dataTexture
		self.dataTexture.replace(region: MTLRegionMake1D(0, self.width * self.height),
														 mipmapLevel: 0,
														 withBytes: UnsafeRawPointer(data),
														 bytesPerRow: 64)
	}
	
	// TODO: Implement transpose in shader or here???
	// TODO: Implement sum() function
	// TODO: Protect each function against incorrect dimensions for arguments
	
	// Overload Operators
	// TODO: Overload the following operators
	// https://www.raywenderlich.com/2271-operator-overloading-in-swift-tutorial
	// * vector(Rx1, 1xC), + vector(Rx1, 1xC), - vector(Rx1, 1xC) (These should be special cases in the normal *, +, and - overloads)
	// +=, -=, ++, --,
	// /, /=, *=,
	// ^
	// ==, !=, <, >, >=, <=
	// @ (matrix multiply), ~ (dot product), exp(Matrix)
	static func *(left:Matrix, right:Matrix) -> Matrix {
		if (left.width == right.width) && (left.height == right.height) {
			return executeSymmetricResultKernel(left: left, right: right, function: self.HadamardKernel!)
		} else if (right.width == 1) && (right.height == left.height) {
			// TODO: Implement (below is incorrect)
			return executeSymmetricResultKernel(left: left, right: right, function: self.HadamardKernel!)
		} else if (right.height == 1) && (right.width == left.width) {
			// TODO: Implement (below is incorrect)
			return executeSymmetricResultKernel(left: left, right: right, function: self.HadamardKernel!)
		} else {
			// TODO: Make errors more descriptive
			print("Argument error between left and right.")
			exit(1)
		}
	}

	static func +(left:Matrix, right:Matrix) -> Matrix {
		if (left.width == right.width) && (left.height == right.height) {
			return executeSymmetricResultKernel(left: left, right: right, function: self.HadamardKernel!)
		} else if (right.width == 1) && (right.height == left.height) {
			// TODO: Implement (below is incorrect)
			return executeSymmetricResultKernel(left: left, right: right, function: self.HadamardKernel!)
		} else if (right.height == 1) && (right.width == left.width) {
			// TODO: Implement (below is incorrect)
			return executeSymmetricResultKernel(left: left, right: right, function: self.HadamardKernel!)
		} else {
			// TODO: Make errors more descriptive
			print("Argument error between left and right.")
			exit(1)
		}
	}

	static func -(left:Matrix, right:Matrix) -> Matrix {
		if (left.width == right.width) && (left.height == right.height) {
			return executeSymmetricResultKernel(left: left, right: right, function: self.HadamardKernel!)
		} else if (right.width == 1) && (right.height == left.height) {
			// TODO: Implement (below is incorrect)
			return executeSymmetricResultKernel(left: left, right: right, function: self.HadamardKernel!)
		} else if (right.height == 1) && (right.width == left.width) {
			// TODO: Implement (below is incorrect)
			return executeSymmetricResultKernel(left: left, right: right, function: self.HadamardKernel!)
		} else {
			// TODO: Make errors more descriptive
			print("Argument error between left and right.")
			exit(1)
		}
	}

	static func executeSymmetricResultKernel(left:Matrix, right:Matrix, function:MTLFunction) -> Matrix {
		let result = Matrix(data:[Float](repeating: 0, count: left.width*left.height),
												width:left.width,
												height:left.height)
		do {
			guard
				let computePipelineState = try Matrix.GPU?.makeComputePipelineState(function: function),
				let commandBuffer = Matrix.CommandQueue?.makeCommandBuffer(),
				let computeEncoder = commandBuffer.makeComputeCommandEncoder()
				else {
					print("Could not create initialize GPU stuff in overloaded *()")
					exit(EXIT_FAILURE)
			}
			computeEncoder.setComputePipelineState(computePipelineState)
			computeEncoder.setTexture(left.dataTexture, index: 0)
			computeEncoder.setTexture(right.dataTexture, index: 1)
			computeEncoder.setTexture(result.dataTexture, index: 2)
			let threadGroupSize = MTLSize(width: 16, height: 1, depth: 1)
			let threadGroupCount = MTLSize(width: 1,
																		 height: 1,
																		 depth: 1)
			computeEncoder.dispatchThreadgroups(threadGroupCount,
																					threadsPerThreadgroup: threadGroupSize)
			computeEncoder.endEncoding()
			commandBuffer.commit()
			commandBuffer.waitUntilCompleted()
		} catch {
			print("Could not initialize ComputePipelineState in overloaded *()")
			exit(EXIT_FAILURE)
		}
		return result
		}
}
