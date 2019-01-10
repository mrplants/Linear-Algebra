//
//  ShaderHelper.h
//  Metal Tutorial 1
//
//  Created by Sean Fitzgerald on 1/9/19.
//  Copyright © 2019 Sean Fitzgerald. All rights reserved.
//

#ifndef ShaderHelper_h
#define ShaderHelper_h

typedef enum : int {
	MatrixArgument1=0,
  MatrixLeftArgument=0,
	MatrixArgument2=1,
	MatrixRightArgument=1,
	MatrixResult,
} TextureIndex;

typedef enum : int {
	ScalarArgument1=0,
	ScalarLeftArgument=0,
	ScalarArgument2=1,
	ScalarRightArgument=1,
	ScalarResult,
	ScalarWidth,
	ScalarHeight,
} BufferIndex;

#endif /* ShaderHelper_h */
