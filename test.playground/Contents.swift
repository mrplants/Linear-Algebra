import Foundation
import MetalPerformanceShaders

let MINI_BATCH_SIZE = 100
let NUM_EPOCHS = 100

// Read the label file and separate between header and body
let label_file_data = try Data(contentsOf: URL(fileURLWithPath: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Data/MNIST/train-labels.idx1-ubyte"))
let label_file_header = label_file_data.subdata(in: 0..<MemoryLayout<Int32>.size*2)
let label_file_body = label_file_data.subdata(in: MemoryLayout<Int32>.size*2..<label_file_data.count)
// Read the header for the number of labels
let number_labels = label_file_header.withUnsafeBytes { (pointer:UnsafePointer<UInt32>) -> Int in
	return Int(CFSwapInt32BigToHost(pointer[1]))
}
// Create a one-hot array to store the labels
// Read the labels
var label_data = UnsafeMutablePointer<Float>(calloc(number_labels*10, MemoryLayout<Float>.size)?.assumingMemoryBound(to: Float.self))!
label_file_body.withUnsafeBytes { (pointer:UnsafePointer<UInt8>) in
	for label_index in 0..<number_labels {
		label_data[label_index*10+Int(pointer[label_index])] = 1
	}
}

// Read the image file and separate between header and body
let image_file_data = try Data(contentsOf: URL(fileURLWithPath: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Data/MNIST/train-images.idx3-ubyte"))
let image_file_header = image_file_data.subdata(in: 0..<MemoryLayout<Int32>.size*4)
let image_file_body = image_file_data.subdata(in: MemoryLayout<Int32>.size*4..<image_file_data.count)
// Read the header for the size of the images
let (_, image_height, image_width) = image_file_header.withUnsafeBytes { (pointer:UnsafePointer<UInt32>) -> (Int, Int, Int) in
	let number_images = Int(CFSwapInt32BigToHost(pointer[1]))
	let image_height = Int(CFSwapInt32BigToHost(pointer[2]))
	let image_width = Int(CFSwapInt32BigToHost(pointer[3]))
	return (number_images, image_width, image_height)
}
// Get the default GPU to create textures
guard let default_GPU = MTLCreateSystemDefaultDevice()
	else {
		print("No GPU available.  Aborting.")
		exit(EXIT_FAILURE)
}

let mini_batch_texture = MPSImage(device: default_GPU,
																	imageDescriptor: MPSImageDescriptor(channelFormat: .unorm8,
																																			width: image_width,
																																			height: image_height,
																																			featureChannels: 1,
																																			numberOfImages: MINI_BATCH_SIZE,
																																			usage: .shaderRead))
func load_mini_batch(mini_batch_index:Int) -> [MPSCNNLossLabels] {
	// Load the images into a GPU texture
	image_file_body.withUnsafeBytes { (pointer:UnsafePointer<UInt8>) in
		for image_index in (mini_batch_index*MINI_BATCH_SIZE)..<((mini_batch_index+1)*MINI_BATCH_SIZE) {
			mini_batch_texture.writeBytes(UnsafeRawPointer(pointer.advanced(by: image_index*image_height*image_width)),
																		dataLayout: .featureChannelsxHeightxWidth,
																		imageIndex: image_index)
		}
	}
	// Load the labels into a GPU state
	var loss_labels = [MPSCNNLossLabels]()
	for label_index in (mini_batch_index*MINI_BATCH_SIZE)..<((mini_batch_index+1)*MINI_BATCH_SIZE) {
		let label_descriptor = MPSCNNLossDataDescriptor(data: Data(bytes: label_data.advanced(by: label_index*10), count: 10*MemoryLayout<Float>.size),
																										layout: .featureChannelsxHeightxWidth,
																										size: MTLSize(width: 1, height: 1, depth: 10))!
		loss_labels.append(MPSCNNLossLabels(device: default_GPU,
																				labelsDescriptor: label_descriptor))
	}
	
	return loss_labels
}

let loss_labels = load_mini_batch(mini_batch_index: 0)

//let image_pointer = UnsafeMutableRawPointer.allocate(byteCount: image_height*image_width, alignment: 1)
//mini_batch_texture.readBytes(image_pointer,
//								dataLayout: .featureChannelsxHeightxWidth,
//								imageIndex: 0)
//let color_space = CGColorSpaceCreateDeviceGray()
//let provider = CGDataProvider(dataInfo: nil, data: image_pointer, size: image_height*image_width, releaseData: {_,_,_ in })!
//let image = CGImage(width: image_width,
//										height: image_height,
//										bitsPerComponent: 8,
//										bitsPerPixel: 8,
//										bytesPerRow: image_width,
//										space: color_space,
//										bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
//										provider: provider,
//										decode: nil,
//										shouldInterpolate: false,
//										intent: .defaultIntent)
//labels[0]

class MyCNNWeights: NSObject, MPSCNNConvolutionDataSource {
	var my_weights:[Float]
	var previous_layer_size:Int
	var input_depth:Int
	var output_depth:Int
	var height:Int
	var width:Int

	init(width:Int, height:Int, input_depth:Int, output_depth:Int, previous_layer_size:Int) {
		self.width = width
		self.height = height
		self.input_depth = input_depth
		self.output_depth = output_depth
		self.previous_layer_size = previous_layer_size
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
	}
	func purge() {
		// Nothing to purge, since the array handles its own memory
	}
	func label() -> String? {
		return "My first CNN filter"
	}
}

// Create the forward pass graph
// INPUT -> CONV -> ReLU -> MAXPOOL -> CONV -> ReLU -> MAXPOOL -> FC -> ReLU -> DROPOUT -> FC -> SOFTMAX
// Compute Loss
// Create the backpropogation graph
// LOSS  -> SOFTMAX -> FC -> DROPOUT -> ReLU -> FC -> MAXPOOL -> ReLU -> CONV -> MAXPOOL -> ReLU -> CONV

func forward_pass_graph() -> [MPSNNFilterNode] {
	let conv1 = MPSCNNConvolutionNode(source: MPSNNImageNode(handle: nil), weights: MyCNNWeights(width: 10,
																																															 height: 10,
																																															 input_depth: 1,
																																															 output_depth: 3,
																																															 previous_layer_size: 28*28))
	let pool1 = MPSCNNPoolingMaxNode(source: conv1.resultImage, filterSize: 2)
	let conv2 = MPSCNNConvolutionNode(source: pool1.resultImage, weights: MyCNNWeights(width: 5,
																																										 height: 5,
																																										 input_depth: 3,
																																										 output_depth: 5,
																																										 previous_layer_size: 19*19*3))
	let pool2 = MPSCNNPoolingMaxNode(source: conv2.resultImage, filterSize: 2)
	let fc1 = MPSCNNFullyConnectedNode(source: pool2.resultImage, weights: MyCNNWeights(width: 15,
																																											height: 15,
																																											input_depth: 5,
																																											output_depth: 15,
																																											previous_layer_size: 15*15*5))
	let fc2 = MPSCNNFullyConnectedNode(source: fc1.resultImage, weights: MyCNNWeights(width: 1,
																																										height: 1,
																																										input_depth: 15,
																																										output_depth: 5,
																																										previous_layer_size: 15))
	return [conv1, pool1, conv2, pool2, fc1, fc2]
}

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

let lossDescriptor = MPSCNNLossDescriptor(type: .categoricalCrossEntropy, reductionType: .mean)
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
	guard let command_buffer = command_queue?.makeCommandBuffer() else { return nil}
	// Encode a batch of images for training
	let labels = load_mini_batch(mini_batch_index: mini_batch_number)
	training_graph.encodeBatch(to: command_buffer,
														 sourceImages: [mini_batch_texture.batchRepresentation()],
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
	for j in 0..<number_labels/MINI_BATCH_SIZE {
		latest_command_buffer = trainingIteration(j);
	}
}





label_data.deallocate()
