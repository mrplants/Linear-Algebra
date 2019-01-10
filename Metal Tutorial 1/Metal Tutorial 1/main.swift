//
//  main.swift
//  Metal Tutorial 1
//
//  Created by Sean Fitzgerald on 1/7/19.
//  Copyright © 2019 Sean Fitzgerald. All rights reserved.
//

import Foundation

// Create the array to multiply
let A = Matrix(data: [2,0,0,0,
											0,2,0,0,
											0,0,2,0,
											0,0,0,2],
							 width: 4,
							 height: 4)
let B = Matrix(data: [1,0,0,0,
											0,2,0,0,
											0,0,3,0,
											0,0,0,4],
							 width: 4,
							 height: 4)

let C = A * B
print("done")
