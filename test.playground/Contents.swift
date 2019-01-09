import Foundation
import MetalPerformanceShaders

let MINI_BATCH_SIZE = 2000
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

var trainImages = try MNISTTrainingImages(filename: "/Users/sean/Library/Mobile Documents/com~apple~CloudDocs/Development/Data/MNIST/train-images.idx3-ubyte")
let image = imageFrom(dataMNIST: trainImages.next()!)
