//
//  main.swift
//  CNN Example
//
//  Created by Sean Fitzgerald on 1/3/19.
//  Copyright © 2019 Sean Fitzgerald. All rights reserved.
//

import Foundation
import MetalPerformanceShaders

let MINI_BATCH_SIZE = 1000
let NUM_EPOCHS = 100


// UTILITY FUNCTIONS
func imageFrom(dataMNIST:Data) -> CGImage? {
	let width = 28
	let height = 28
	let image_pointer = UnsafeMutableRawPointer.allocate(byteCount: width * height, alignment: 1)
	_ = dataMNIST.withUnsafeBytes({(pointer:UnsafePointer<UInt8>) in
		memcpy(image_pointer, pointer, width * height)
	})
	return CGImage(width: width,
								 height: height,
								 bitsPerComponent: 8,
								 bitsPerPixel: 8,
								 bytesPerRow: width,
								 space: CGColorSpaceCreateDeviceGray(),
								 bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
								 provider: CGDataProvider(dataInfo: nil,
																					data: image_pointer,
																					size: width * height,
																					releaseData: { (info:UnsafeMutableRawPointer?, data:UnsafeRawPointer, size:Int) in
																						data.deallocate()
								})!,
								 decode: nil,
								 shouldInterpolate: false,
								 intent: .defaultIntent)
}

// STORAGE ACCESS
struct MNISTTrainingLabels : Sequence, IteratorProtocol {
	var m:Int
	var labels:[Int]
	var index = 0
	
	// Initialize with the location of the MNIST data file
	init(filename:String) throws {
		// Read the label file and separate between header and body
		let labelFileData = try Data(contentsOf: URL(fileURLWithPath: filename))
		let labelFileHeader = labelFileData.subdata(in: 0..<MemoryLayout<Int32>.size*2)
		// Read the header for the number of labels
		self.m = labelFileHeader.withUnsafeBytes { (pointer:UnsafePointer<UInt32>) -> Int in
			return Int(CFSwapInt32BigToHost(pointer[1]))
		}
		// Get the labels
		self.labels = [Int](repeating: 0, count: self.m)
		loadLabels(fileBodyData: labelFileData.subdata(in: MemoryLayout<Int32>.size*2..<labelFileData.count))
	}
	// Create one-hot labels from the input data
	mutating func loadLabels(fileBodyData:Data) {
		// Set the labels array to all zeros
		fileBodyData.withUnsafeBytes { (pointer:UnsafePointer<UInt8>) in
			for labelIndex in 0..<self.m {
				self.labels[labelIndex] = Int(pointer[labelIndex])
			}
		}
	}
	// Returns the next training pair
	mutating func next() -> [Float]? {
		// Construct the one-hot label array
		var oneHotLabels = [Float](repeating: 0, count: 10)
		oneHotLabels[self.labels[index]] = 1.0
		return oneHotLabels
	}
}

struct MNISTTrainingImages : Sequence, IteratorProtocol {
	var m:Int
	var width:Int
	var height:Int
	var images:Data
	var index = 0

	// Initialize with the location of the MNIST data file
	init(filename:String) throws {
		let image_file_data = try Data(contentsOf: URL(fileURLWithPath: filename))
		let image_file_header = image_file_data.subdata(in: 0..<MemoryLayout<Int32>.size*4)
		// Read the header for the size and size of the images
		(self.m, self.height, self.width) = image_file_header.withUnsafeBytes { (pointer:UnsafePointer<UInt32>) -> (Int, Int, Int) in
			let number_images = Int(CFSwapInt32BigToHost(pointer[1]))
			let image_height = Int(CFSwapInt32BigToHost(pointer[2]))
			let image_width = Int(CFSwapInt32BigToHost(pointer[3]))
			return (number_images, image_width, image_height)
		}
		self.images = image_file_data.subdata(in: MemoryLayout<Int32>.size*4..<image_file_data.count)
	}
	mutating func next() -> Data? {
		let image = self.images[self.index*self.height*self.width..<(self.index+1)*self.height*self.width]
		self.index += 1
		return image
	}
}

struct MNISTMiniBatches : Sequence, IteratorProtocol {
	var trainingLabels:MNISTTrainingLabels
	var trainingImages:MNISTTrainingImages
	var gpu:MTLDevice
	var miniBatchSize:Int
	
	init(gpu:MTLDevice, miniBatchSize:Int) throws {
		self.trainingLabels = try MNISTTrainingLabels(filename: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Data/MNIST/train-labels.idx1-ubyte")
		self.trainingImages = try MNISTTrainingImages(filename: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Data/MNIST/train-images.idx3-ubyte")
		self.gpu = gpu
		self.miniBatchSize = miniBatchSize
	}
	
	mutating func reset() throws {
		self.trainingLabels = try MNISTTrainingLabels(filename: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Data/MNIST/train-labels.idx1-ubyte")
		self.trainingImages = try MNISTTrainingImages(filename: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Data/MNIST/train-images.idx3-ubyte")
	}
	
	// Get the next training mini-batch
	mutating func next() -> ([MPSCNNLossLabels], [MPSImage])? {
		var images = [MPSImage]()
		var labels = [MPSCNNLossLabels]()
		for _ in 0..<self.miniBatchSize {
			// Load the images into a GPU texture
			self.trainingImages.next()?.withUnsafeBytes {(pointer:UnsafePointer<UInt8>) in
				let image = MPSImage(device: self.gpu,
														 imageDescriptor: MPSImageDescriptor(channelFormat: .unorm8,
																																 width: self.trainingImages.width,
																																 height: self.trainingImages.height,
																																 featureChannels: 1))
				image.writeBytes(pointer,
												 dataLayout: .HeightxWidthxFeatureChannels,
												 imageIndex: 0)
				images.append(image)
			}
			labels.append(MPSCNNLossLabels(device: self.gpu,
																		 labelsDescriptor: MPSCNNLossDataDescriptor(data: Data(bytes: UnsafeRawPointer(self.trainingLabels.next()!),
																																													 count: MemoryLayout<Float>.size*10),
																																								layout: .HeightxWidthxFeatureChannels,
																																								size: MTLSize(width: 1, height: 1, depth: 10))!))
		}
		return (labels, images)
	}
}

class MyCNNWeights: NSObject, MPSCNNConvolutionDataSource {
	var my_weights:[Float]
	var previous_layer_size:Int
	var input_depth:Int
	var output_depth:Int
	var height:Int
	var width:Int
	var optimizer:MPSNNOptimizerStochasticGradientDescent
	var gpu:MTLDevice
	
	init(gpu:MTLDevice, width:Int, height:Int, input_depth:Int, output_depth:Int, previous_layer_size:Int) {
		self.width = width
		self.height = height
		self.input_depth = input_depth
		self.output_depth = output_depth
		self.previous_layer_size = previous_layer_size
		self.gpu = gpu
		self.optimizer = MPSNNOptimizerStochasticGradientDescent(device: gpu, learningRate: 0.01)
		self.my_weights = (0...width*height*output_depth).map { _ in Float.random(in: 0...1)*sqrtf(Float(2.0)/Float(previous_layer_size)) }
	}
	func copy(with zone: NSZone? = nil) -> Any {
		let my_copy = MyCNNWeights(gpu:self.gpu,
															 width: self.width,
															 height: self.height,
															 input_depth: self.input_depth,
															 output_depth: self.output_depth,
															 previous_layer_size: self.previous_layer_size)
		my_copy.my_weights = self.my_weights
		return my_copy
	}
	func dataType() -> MPSDataType {
		return MPSDataType.float16
	}
	func descriptor() -> MPSCNNConvolutionDescriptor {
		return MPSCNNConvolutionDescriptor(kernelWidth: self.width,
																			 kernelHeight: self.height,
																			 inputFeatureChannels: self.input_depth,
																			 outputFeatureChannels: self.output_depth)
	}
	func weights() -> UnsafeMutableRawPointer {
		return UnsafeMutableRawPointer(&self.my_weights)
	}
	func biasTerms() -> UnsafeMutablePointer<Float>? {
		return nil
	}
	func load() -> Bool {
		// He Initialization
		self.my_weights = (0...self.width*self.height*self.output_depth).map { _ in Float.random(in: 0...1)*sqrtf(Float(2.0)/Float(self.previous_layer_size)) }
		return true
	}
	func purge() {
		// Nothing to purge, since the array handles its own memory
	}
	func label() -> String? {
		return "My NN filter"
	}
	func update(with commandBuffer: MTLCommandBuffer,
							gradientState: MPSCNNConvolutionGradientState,
							sourceState: MPSCNNConvolutionWeightsAndBiasesState) -> MPSCNNConvolutionWeightsAndBiasesState? {
		// TODO: This seems like a sketchy way around readCount.
		sourceState.readCount += 1
		self.optimizer.encode(commandBuffer: commandBuffer,
													convolutionGradientState: gradientState,
													convolutionSourceState: sourceState,
													inputMomentumVectors: nil,
													resultState: sourceState)
		return sourceState
	}
}

// Create the forward pass graph
// INPUT -> CONV -> ReLU -> MAXPOOL -> CONV -> ReLU -> MAXPOOL -> FC -> ReLU -> DROPOUT -> FC -> SOFTMAX
// Compute Loss
// Create the backpropogation graph
// LOSS  -> SOFTMAX -> FC -> DROPOUT -> ReLU -> FC -> MAXPOOL -> ReLU -> CONV -> MAXPOOL -> ReLU -> CONV

func forward_pass_graph(gpu:MTLDevice) -> [MPSNNFilterNode] {
//	let conv1 = MPSCNNConvolutionNode(source: MPSNNImageNode(handle: nil), weights: MyCNNWeights(gpu:gpu,
//																																															 width: 9,
//																																															 height: 9,
//																																															 input_depth: 1,
//																																															 output_depth: 5,
//																																															 previous_layer_size: 28*28*1))
//	let relu1 = MPSCNNNeuronReLUNode(source: conv1.resultImage)
//	let pool1 = MPSCNNPoolingMaxNode(source: relu1.resultImage, filterSize: 2)
//	let conv2 = MPSCNNConvolutionNode(source: pool1.resultImage, weights: MyCNNWeights(gpu:gpu,
//																																										 width: 5,
//																																										 height: 5,
//																																										 input_depth: 5,
//																																										 output_depth: 5,
//																																										 previous_layer_size: 10*10*5))
//	let relu2 = MPSCNNNeuronReLUNode(source: conv2.resultImage)
//	let pool2 = MPSCNNPoolingMaxNode(source: relu2.resultImage, filterSize: 2)
	let fc1 = MPSCNNFullyConnectedNode(source: MPSNNImageNode(handle: nil)/*pool2.resultImage*/, weights: MyCNNWeights(gpu:gpu,
																																											width: 28,
																																											height: 28,
																																											input_depth: 1,
																																											output_depth: 300,
																																											previous_layer_size: 28*28))
	let relu3 = MPSCNNNeuronReLUNode(source: fc1.resultImage)
	let fc2 = MPSCNNFullyConnectedNode(source: relu3.resultImage, weights: MyCNNWeights(gpu:gpu,
																																											width: 1,
																																											height: 1,
																																											input_depth: 300,
																																											output_depth: 10,
																																											previous_layer_size: 300))
	let softmax = MPSCNNSoftMaxNode(source: fc2.resultImage)
	return [/*conv1, relu1, pool1, conv2, relu2, pool2,*/ fc1, relu3, fc2, softmax]
}
let lossDescriptor = MPSCNNLossDescriptor(type: .softMaxCrossEntropy, reductionType: .mean)
lossDescriptor.numberOfClasses = 10
func backprop_graph(forward_graph:[MPSNNFilterNode]) -> [MPSNNFilterNode] {
	var return_graph = forward_graph
	return_graph.append(MPSCNNLossNode(source: forward_graph.last!.resultImage, lossDescriptor: lossDescriptor))
	// Backpropogate along the forward prop graph
	for filter_node in forward_graph.reversed() {
		return_graph.append(filter_node.gradientFilter(withSource: return_graph.last!.resultImage))
	}
	return return_graph
}
func make_inference_graph(gpu:MTLDevice) -> MPSNNImageNode {
	let forward_graph = forward_pass_graph(gpu: gpu)
	return forward_graph.last!.resultImage
}
func make_training_graph(gpu:MTLDevice) -> MPSNNImageNode {
	let training_graph = backprop_graph(forward_graph: forward_pass_graph(gpu: gpu))
	return training_graph.last!.resultImage
}

// Get the default GPU to create textures
let default_GPU = MTLCreateSystemDefaultDevice()!

// Retrieve training data
var trainingMiniBatches = try MNISTMiniBatches(gpu: default_GPU, miniBatchSize: MINI_BATCH_SIZE)
let training_graph = MPSNNGraph(device: default_GPU, resultImage: make_training_graph(gpu: default_GPU), resultImageIsNeeded: false)!
let command_queue = default_GPU.makeCommandQueue()

// Execute Graph in a Training Loop with Double Buffering
//let doubleBufferSemaphore = DispatchSemaphore(value: 1)
func trainingIteration() throws {
//	_ = doubleBufferSemaphore.wait(timeout: .distantFuture)
	guard let command_buffer = command_queue?.makeCommandBufferWithUnretainedReferences() else { return }
	// Encode a batch of images for training
	guard let (labels, images) = trainingMiniBatches.next() else { return }
	training_graph.encodeBatch(to: command_buffer,
														 sourceImages: [images],
														 sourceStates: [labels])
//	command_buffer.addCompletedHandler { commandBuffer in
//		// Callback is called when GPU is done executing the graph (outputBatch is ready)
//		doubleBufferSemaphore.signal()
//	}
	command_buffer.commit()
	command_buffer.waitUntilCompleted()
}

var latest_command_buffer:MTLCommandBuffer? = nil

// NUM_EPOCHS is the number of times we iterate over an entire dataset
// NUM_ITERATIONS_PER_EPOCH is the number of images in a dataset, divided by batch size
for i in 0..<NUM_EPOCHS {
	for j in 0..<60000/MINI_BATCH_SIZE {
		try trainingIteration();
		if j % 10 == 0 { print("mini-batch finished " + String(j)) }
	}
	// TODO: Print/Save loss and accuracy
	// TODO: Time the epochs
//	latest_command_buffer!.waitUntilCompleted()
	print("Completed epoch number " + String(i))
	try trainingMiniBatches.reset()
}

