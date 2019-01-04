import Foundation
import MetalPerformanceShaders

let MINI_BATCH_SIZE = 1000
let NUM_EPOCHS = 100

// UTILITY FUNCTIONS
func imageFrom(dataMNIST:Data) -> CGImage? {
	let width = 28
	let height = 28
	let image_pointer = UnsafeMutableRawPointer.allocate(byteCount: width * height, alignment: 1)
	dataMNIST.withUnsafeBytes({(pointer:UnsafePointer<UInt8>) in
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
	var labels:[Float]?
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
		loadOneHotLabels(fileBodyData: labelFileData.subdata(in: MemoryLayout<Int32>.size*2..<labelFileData.count))
	}
	// Create one-hot labels from the input data
	mutating func loadOneHotLabels(fileBodyData:Data) {
		// Set the labels array to all zeros
		self.labels = [Float].init(repeating: 0, count: self.m * 10)
		fileBodyData.withUnsafeBytes { (pointer:UnsafePointer<UInt8>) in
			for labelIndex in 0..<self.m {
				// Apply one-hot encoding based on the label data
				self.labels?[labelIndex*10+Int(pointer[labelIndex])] = 1
			}
		}
	}
	// Returns the next training pair
	mutating func next() -> MPSCNNLossDataDescriptor? {
		// Construct the one-hot label array
		guard var labels = self.labels else { return nil }
		let lossDescriptor = MPSCNNLossDataDescriptor(data: Data(bytesNoCopy: &(labels[(self.index*10)..<((self.index+1)*10)]),
																														 count: 10 * MemoryLayout<Float>.size,
																														 deallocator: .none),
																									layout: .featureChannelsxHeightxWidth,
																									size: MTLSize(width: 1, height: 1, depth: 10))
		self.index += 1
		return lossDescriptor
	}
	func currentOneHotLabel() -> [Float]? {
		return self.labels?[((self.index-1)*10)..<(self.index*10)]
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
		let image = self.images.subdata(in: self.index*self.height*self.width..<(self.index+1)*self.height*self.width)
		self.index += 1
		return image
	}
}

struct MNISTMiniBatches : Sequence, IteratorProtocol {
	var trainingLabels:MNISTTrainingLabels
	var trainingImages:MNISTTrainingImages
	var gpu:MTLDevice
	var miniBatchSize:Int
	var textures:MPSImage
	var m:Int
	var imageWidth:Int
	var imageHeight:Int
	
	// TODO: State is saved in the RAM.  Create a minibatch of label data once and then continue to reference it with the same loss data descriptor.
	
	init(gpu:MTLDevice, miniBatchSize:Int) throws {
		self.trainingLabels = try MNISTTrainingLabels(filename: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Data/MNIST/train-labels.idx1-ubyte")
		self.trainingImages = try MNISTTrainingImages(filename: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Data/MNIST/train-images.idx3-ubyte")
		self.m = self.trainingImages.m
		self.imageWidth = self.trainingImages.width
		self.imageHeight = self.trainingImages.height
		self.gpu = gpu
		self.miniBatchSize = miniBatchSize
		self.textures = MPSImage(device: self.gpu,
														 imageDescriptor: MPSImageDescriptor(channelFormat: .unorm8,
																																 width: self.trainingImages.width,
																																 height: self.trainingImages.height,
																																 featureChannels: 1,
																																 numberOfImages: self.miniBatchSize,
																																 usage: .shaderRead))
	}
	
	// Get the next training mini-batch
	mutating func next() -> ([MPSCNNLossLabels], [MPSImage])? {
		var lossLabels = [MPSCNNLossLabels]()
		for miniBatchIndex in 0..<self.miniBatchSize {
			// Load the images into a GPU texture
			self.trainingImages.next()?.withUnsafeBytes {(pointer:UnsafePointer<UInt8>) in
				self.textures.writeBytes(pointer,
																 dataLayout: .featureChannelsxHeightxWidth,
																 imageIndex: miniBatchIndex)
			}
			
			// Create and load the GPU loss labels
			lossLabels.append(MPSCNNLossLabels(device: self.gpu,
																				 labelsDescriptor: self.trainingLabels.next()!))
		}
		return (lossLabels, self.textures.batchRepresentation())
	}
}

// Get the default GPU to create textures
let default_GPU = MTLCreateSystemDefaultDevice()!

// Retrieve training data
var trainingMiniBatches = try MNISTMiniBatches(gpu: default_GPU, miniBatchSize: MINI_BATCH_SIZE)



class MyCNNWeights: NSObject, MPSCNNConvolutionDataSource {
	var my_weights:[Float]
	var previous_layer_size:Int
	var input_depth:Int
	var output_depth:Int
	var height:Int
	var width:Int
	var optimizer:MPSNNOptimizerStochasticGradientDescent
	
	init(width:Int, height:Int, input_depth:Int, output_depth:Int, previous_layer_size:Int) {
		self.width = width
		self.height = height
		self.input_depth = input_depth
		self.output_depth = output_depth
		self.previous_layer_size = previous_layer_size
		self.optimizer = MPSNNOptimizerStochasticGradientDescent(device: default_GPU, learningRate: 0.01)
		self.my_weights = (0...width*height*output_depth).map { _ in Float.random(in: 0...1)*sqrtf(Float(2.0)/Float(previous_layer_size)) }
	}
	func copy(with zone: NSZone? = nil) -> Any {
		let my_copy = MyCNNWeights(width: self.width,
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

func forward_pass_graph() -> [MPSNNFilterNode] {
	let conv1 = MPSCNNConvolutionNode(source: MPSNNImageNode(handle: nil), weights: MyCNNWeights(width: 9,
																																															 height: 9,
																																															 input_depth: 1,
																																															 output_depth: 3,
																																															 previous_layer_size: 28*28*1))
	let pool1 = MPSCNNPoolingMaxNode(source: conv1.resultImage, filterSize: 2)
	let conv2 = MPSCNNConvolutionNode(source: pool1.resultImage, weights: MyCNNWeights(width: 5,
																																										 height: 5,
																																										 input_depth: 3,
																																										 output_depth: 5,
																																										 previous_layer_size: 10*10*3))
	let pool2 = MPSCNNPoolingMaxNode(source: conv2.resultImage, filterSize: 2)
	let fc1 = MPSCNNFullyConnectedNode(source: pool2.resultImage, weights: MyCNNWeights(width: 3,
																																											height: 3,
																																											input_depth: 5,
																																											output_depth: 15,
																																											previous_layer_size: 3*3*5))
	let fc2 = MPSCNNFullyConnectedNode(source: fc1.resultImage, weights: MyCNNWeights(width: 1,
																																										height: 1,
																																										input_depth: 15,
																																										output_depth: 10,
																																										previous_layer_size: 15))
	// TODO: Add softmax output
	// TODO: Add ReLU neurons
	return [conv1, pool1, conv2, pool2, fc1, fc2]
}
let lossDescriptor = MPSCNNLossDescriptor(type: .categoricalCrossEntropy, reductionType: .mean)
func backprop_graph(forward_graph:[MPSNNFilterNode]) -> [MPSNNFilterNode] {
	var return_graph = forward_graph
	return_graph.append(MPSCNNLossNode(source: forward_graph.last!.resultImage, lossDescriptor: lossDescriptor))
	// Backpropogate along the forward prop graph
	for filter_node in forward_graph.reversed() {
		return_graph.append(filter_node.gradientFilter(withSource: return_graph.last!.resultImage))
	}
	return return_graph
}
func make_inference_graph() -> MPSNNImageNode {
	let forward_graph = forward_pass_graph()
	return forward_graph.last!.resultImage
}
func make_training_graph() -> MPSNNImageNode {
	let training_graph = backprop_graph(forward_graph: forward_pass_graph())
	return training_graph.last!.resultImage
}

let command_queue = default_GPU.makeCommandQueue()
let training_graph = MPSNNGraph(device: default_GPU, resultImage: make_training_graph(), resultImageIsNeeded: false)!

// Execute Graph in a Training Loop with Double Buffering
let doubleBufferSemaphore = DispatchSemaphore(value: 2)
func trainingIteration(_ mini_batch_number:Int) -> MTLCommandBuffer? {
	doubleBufferSemaphore.wait(timeout: .distantFuture)
	guard let command_buffer = command_queue?.makeCommandBuffer() else { return nil }
	// Encode a batch of images for training
	print("about to load mini-batch")
	guard let (labels, images) = trainingMiniBatches.next() else { return nil }
	print("mini-batch loaded")
	training_graph.encodeBatch(to: command_buffer,
														 sourceImages: [images],
														 sourceStates: [labels])
	command_buffer.addCompletedHandler { commandBuffer in
		// Callback is called when GPU is done executing the graph (outputBatch is ready)
		doubleBufferSemaphore.signal()
	}
	command_buffer.commit()
	return command_buffer
}

var latest_command_buffer:MTLCommandBuffer? = nil

// NUM_EPOCHS is the number of times we iterate over an entire dataset
// NUM_ITERATIONS_PER_EPOCH is the number of images in a dataset, divided by batch size
for i in 0..<NUM_EPOCHS {
	for j in 0..<trainingMiniBatches.m/MINI_BATCH_SIZE {
		latest_command_buffer = trainingIteration(j);
		print("mini-batch finished")
	}
	// TODO: Print/Save loss and accuracy
	// TODO: Time the epochs
	latest_command_buffer!.waitUntilCompleted()
	print("Completed epoch number " + String(i))
}

