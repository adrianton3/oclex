package image;

import static org.lwjgl.opencl.CL10.CL_DEVICE_TYPE_GPU;
import static org.lwjgl.opencl.CL10.CL_MEM_READ_ONLY;
import static org.lwjgl.opencl.CL10.CL_MEM_READ_WRITE;
import static org.lwjgl.opencl.CL10.CL_MEM_USE_HOST_PTR;
import static org.lwjgl.opencl.CL10.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
import static org.lwjgl.opencl.CL10.CL_QUEUE_PROFILING_ENABLE;
import static org.lwjgl.opencl.CL10.CL_RGBA;
import static org.lwjgl.opencl.CL10.CL_UNSIGNED_INT8;
import static org.lwjgl.opencl.CL10.clBuildProgram;
import static org.lwjgl.opencl.CL10.clCreateCommandQueue;
import static org.lwjgl.opencl.CL10.clCreateKernel;
import static org.lwjgl.opencl.CL10.clCreateProgramWithSource;
import static org.lwjgl.opencl.CL10.clEnqueueNDRangeKernel;
import static org.lwjgl.opencl.CL10.clEnqueueReadImage;
import static org.lwjgl.opencl.CL10.clReleaseCommandQueue;
import static org.lwjgl.opencl.CL10.clReleaseContext;
import static org.lwjgl.opencl.CL10.clReleaseKernel;
import static org.lwjgl.opencl.CL10.clReleaseMemObject;
import static org.lwjgl.opencl.CL10.clReleaseProgram;

import java.awt.BorderLayout;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;

import org.lwjgl.BufferUtils;
import org.lwjgl.LWJGLException;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLCommandQueue;
import org.lwjgl.opencl.CLContext;
import org.lwjgl.opencl.CLDevice;
import org.lwjgl.opencl.CLKernel;
import org.lwjgl.opencl.CLMem;
import org.lwjgl.opencl.CLPlatform;
import org.lwjgl.opencl.CLProgram;
import org.lwjgl.opencl.Util;
import org.lwjgl.opencl.api.CLImageFormat;

public class SimpleImageDemo {
	CLPlatform platform;
	List<CLDevice> devices;
	CLDevice device;
	CLContext context;
	CLCommandQueue queue;
	
	CLMem inputImageMem, outputImageMem;
	
	CLProgram program;
	CLKernel kernel;

	final int imageSizeX, imageSizeY;

	SimpleImageDemo() throws LWJGLException {
		// input and output images
		BufferedImage inputImage = createBufferedImage("lena512color.png");
		imageSizeX = inputImage.getWidth();
		imageSizeY = inputImage.getHeight();

		BufferedImage outputImage = new BufferedImage(imageSizeX, imageSizeY, BufferedImage.TYPE_INT_RGB);
		
		// panel
		JPanel mainPanel = new JPanel(new GridLayout(0, 2));
		
		JLabel inputLabel = new JLabel(new ImageIcon(inputImage));
		JLabel outputLabel = new JLabel(new ImageIcon(outputImage));
		
		mainPanel.add(inputLabel, BorderLayout.CENTER);
		mainPanel.add(outputLabel, BorderLayout.CENTER);

		// frame
		JFrame frame = new JFrame("OpenCL Simple Image Manipulation Demo");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setLayout(new BorderLayout());
		frame.add(mainPanel, BorderLayout.CENTER);
		frame.pack();
		frame.setVisible(true);

		// launch
		initCL();
		initMemory(inputImage);
		processImage(outputImage);
		freeCL();
	}
	
	String packKernel() {
		return KernelBuilder.create();
	}

	void initCL() throws LWJGLException {
		// initialization
		CL.create();
		platform = CLPlatform.getPlatforms().get(0);

		devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
		context = CLContext.create(platform, devices, null, null, null);
		device = devices.get(0);
		
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, null);

		// kernel creation
		program = clCreateProgramWithSource(context, packKernel(), null);
		Util.checkCLError(clBuildProgram(program, device, "", null));
		kernel = clCreateKernel(program, "im", null);
	}

	void initMemory(BufferedImage inputImage) {
		DataBufferInt dataBufferSrc = (DataBufferInt)inputImage.getRaster().getDataBuffer();
		ByteBuffer image = BufferUtils.createByteBuffer(dataBufferSrc.getSize() * 4);
		for(int p: dataBufferSrc.getData()) {
			image.put((byte) (0xFF));
			image.put((byte) ((p & 0x00FF0000) >> 16));
			image.put((byte) ((p & 0x0000FF00) >> 8));
			image.put((byte) ((p & 0x000000FF)));
		}
		image.rewind();

		CLImageFormat imageFormat = new CLImageFormat(CL_RGBA, CL_UNSIGNED_INT8);
		
		inputImageMem = CLMem.createImage2D(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, imageFormat,
				imageSizeX, imageSizeY, imageSizeX * 4, image, null);
		outputImageMem = CLMem.createImage2D(context, CL_MEM_READ_WRITE, imageFormat, imageSizeX, imageSizeY, 0,
				null, null);
	}

	void processImage(BufferedImage outputImage) {
		// set up workSize
		PointerBuffer kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(2);
		kernel1DGlobalWorkSize.put(0, imageSizeX);
		kernel1DGlobalWorkSize.put(1, imageSizeY);

		kernel.setArg(0, inputImageMem);
		kernel.setArg(1, outputImageMem);

		// launch
		clEnqueueNDRangeKernel(queue, kernel, 2, null, kernel1DGlobalWorkSize, null, null, null);

		// prepare read
		DataBufferInt dataBufferDest = (DataBufferInt)outputImage.getRaster().getDataBuffer();
		ByteBuffer image = BufferUtils.createByteBuffer(dataBufferDest.getSize() * 4);
		PointerBuffer region = PointerBuffer.allocateDirect(3);
		region.put(imageSizeX);
		region.put(imageSizeY);
		region.put(1);
		region.rewind();
		PointerBuffer origin = PointerBuffer.allocateDirect(3);
		origin.put(0);
		origin.put(0);
		origin.put(0);
		origin.rewind();

		// request read
		clEnqueueReadImage(queue, outputImageMem, 1, origin, region, 0, 0, image, null, null);

		// put data back into image
		image.rewind();
		for(int i = 0; i < image.capacity() / 4; i++) {
			int col = 0;
			col += image.get() << 24;
			col += image.get() << 16;
			col += image.get() << 8;
			col += image.get();
			dataBufferDest.setElem(i, col);
		}
	}

	void freeCL() {
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseMemObject(outputImageMem);
		clReleaseMemObject(inputImageMem);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		CL.destroy();
	}

	static BufferedImage createBufferedImage(String fileName) {
		BufferedImage image = null;
		try {
			image = ImageIO.read(new File(fileName));
		} catch(IOException e) {
			e.printStackTrace();
			return null;
		}

		BufferedImage result = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
		Graphics g = result.createGraphics();
		g.drawImage(image, 0, 0, null);
		g.dispose();
		return result;
	}
	
	public static void main(String args[]) {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				try {
					new SimpleImageDemo();
				} catch(LWJGLException e) {
					e.printStackTrace();
				}
			}
		});
	}
}