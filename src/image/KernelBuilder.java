package image;

public class KernelBuilder {
 static String create() {
 	final String kernelSource =
 		"const sampler_t samplerIn = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; " +
 		"const sampler_t samplerOut = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; " +
 		"\n" +
 		"kernel void im(read_only image2d_t inputImage, write_only image2d_t targetImage) {" +
 		" int gidX = get_global_id(0); " +
 		" int gidY = get_global_id(1); " +
 		"\n" +
 		" int2 posIn = { gidX, gidY }; " +
 		" uint4 pixel = read_imageui(inputImage, samplerIn, posIn); " +
 		" pixel.y = gidX ^ gidY; " +
 		" write_imageui(targetImage, posIn, pixel); " +
 		"}";
 	return kernelSource;
 }
}
