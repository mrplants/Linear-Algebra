//
//  main.swift
//  Metal Tutorial 1
//
//  Created by Sean Fitzgerald on 1/7/19.
//  Copyright © 2019 Sean Fitzgerald. All rights reserved.
//

import Foundation
import Metal
import MetalKit

guard
	let gpu = MTLCreateSystemDefaultDevice(),
	let commandQueue = gpu.makeCommandQueue(),
	let commandBuffer = commandQueue.makeCommandBuffer(),
	let defaultLibrary = gpu.makeDefaultLibrary(),
	let kernelFunction = defaultLibrary.makeFunction(name: "grayscale")
else {exit(1)}

let imageURL = URL(fileURLWithPath: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Software/Artificial Intelligence/CNN Example Swift/Metal Tutorial 1/Metal Tutorial 1/Lenna.png")
let lennaTexture = try MTKTextureLoader(device: gpu).newTexture(URL: imageURL, options: nil)
let texDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm,
																														 width: lennaTexture.width,
																														 height: lennaTexture.height,
																														 mipmapped: false)
texDescriptor.usage = .shaderWrite
let outputTexture = gpu.makeTexture(descriptor: texDescriptor)

let computePipelineState = try gpu.makeComputePipelineState(function: kernelFunction)
let computeEncoder = commandBuffer.makeComputeCommandEncoder()
computeEncoder?.setComputePipelineState(computePipelineState)
computeEncoder?.setTexture(lennaTexture, index: 0)
computeEncoder?.setTexture(outputTexture, index: 1)
let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
let threadGroupCount = MTLSize(width: (lennaTexture.width + threadGroupSize.width - 1) / threadGroupSize.width,
															 height: (lennaTexture.height + threadGroupSize.height - 1) / threadGroupSize.height,
															 depth: 1)
computeEncoder?.dispatchThreadgroups(threadGroupCount,
																		 threadsPerThreadgroup: threadGroupSize)
computeEncoder?.endEncoding()
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

print("done")
