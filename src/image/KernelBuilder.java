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
 		" int2 posIn0 = { gidX - 1, gidY }; " +
 		" int2 posIn1 = { gidX, gidY + 1 }; " +
 		" int2 posIn2 = { gidX + 1, gidY }; " +
 		" int2 posIn3 = { gidX, gidY - 1 }; " +
		" int2 posIno = { gidX, gidY }; " +
		"\n" +
		" uint4 pixel0 = read_imageui(inputImage, samplerIn, posIn0); " +
		" uint4 pixel1 = read_imageui(inputImage, samplerIn, posIn1); " +
		" uint4 pixel2 = read_imageui(inputImage, samplerIn, posIn2); " +
		" uint4 pixel3 = read_imageui(inputImage, samplerIn, posIn3); " +
 		" uint4 pixelo = read_imageui(inputImage, samplerIn, posIno); " +
 		"\n" +
 		" pixelo = (abs(pixel0 - pixelo) + abs(pixel1 - pixelo) + abs(pixel2 - pixelo) + abs(pixel3 - pixelo)) / 4; " +
 		" write_imageui(targetImage, posIno, pixelo); " +
 		"}";
 	return kernelSource;
 }
}
