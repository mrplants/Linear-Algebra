//
//  Matrix.swift
//  Metal Tutorial 1
//
//  Created by Sean Fitzgerald on 1/8/19.
//  Copyright © 2019 Sean Fitzgerald. All rights reserved.
//

import Foundation
import Metal

struct Matrix {
	// Static Properties
	static let GPU = MTLCreateSystemDefaultDevice()
	static let CommandQueue = Matrix.GPU?.makeCommandQueue()
	static let DefaultLibrary = Matrix.GPU?.makeDefaultLibrary()
	static let HadamardKernel = Matrix.DefaultLibrary?.makeFunction(name: "hadamardProductKernel")
	
	let dataTexture:MTLTexture
	let width:Int
	let height:Int
	
	init(data:[Float], width:Int, height:Int) {
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
	
	static func *(left:Matrix, right:Matrix) -> Matrix {
		let result = Matrix(data:[Float](repeating: 0, count: left.width*left.height),
												width:left.width,
												height:left.height)
		do {
			guard
				let computePipelineState = try Matrix.GPU?.makeComputePipelineState(function: Matrix.HadamardKernel!),
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
