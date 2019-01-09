//
//  grayscale.metal
//  Metal Tutorial 1
//
//  Created by Sean Fitzgerald on 1/7/19.
//  Copyright © 2019 Sean Fitzgerald. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

// Rec 709 LUMA values for grayscale image conversion
constant half3 kRec709Luma = half3(0.2126, 0.7152, 0.0722);

kernel void
grayscale(texture2d<half, access::read>  inTexture  [[texture(0)]],
								texture2d<half, access::write> outTexture [[texture(1)]],
								uint2                          gid         [[thread_position_in_grid]])
{
	half4 inColor  = inTexture.read(gid);
	half  gray     = dot(inColor.rgb, kRec709Luma);
	outTexture.write(half4(gray, gray, gray, 1.0), gid);
}
