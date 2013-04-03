package mru;

import static org.lwjgl.opencl.CL10.CL_DEVICE_TYPE_GPU;
import static org.lwjgl.opencl.CL10.CL_MEM_COPY_HOST_PTR;
import static org.lwjgl.opencl.CL10.CL_MEM_READ_WRITE;
import static org.lwjgl.opencl.CL10.CL_QUEUE_PROFILING_ENABLE;
import static org.lwjgl.opencl.CL10.clBuildProgram;
import static org.lwjgl.opencl.CL10.clCreateBuffer;
import static org.lwjgl.opencl.CL10.clCreateCommandQueue;
import static org.lwjgl.opencl.CL10.clCreateKernel;
import static org.lwjgl.opencl.CL10.clCreateProgramWithSource;
import static org.lwjgl.opencl.CL10.clEnqueueNDRangeKernel;
import static org.lwjgl.opencl.CL10.clEnqueueReadBuffer;
import static org.lwjgl.opencl.CL10.clEnqueueWriteBuffer;
import static org.lwjgl.opencl.CL10.clFinish;
import static org.lwjgl.opencl.CL10.clReleaseCommandQueue;
import static org.lwjgl.opencl.CL10.clReleaseContext;
import static org.lwjgl.opencl.CL10.clReleaseKernel;
import static org.lwjgl.opencl.CL10.clReleaseMemObject;
import static org.lwjgl.opencl.CL10.clReleaseProgram;
import static org.lwjgl.opengl.GL11.GL_COLOR_BUFFER_BIT;
import static org.lwjgl.opengl.GL11.GL_MODELVIEW;
import static org.lwjgl.opengl.GL11.GL_PROJECTION;
import static org.lwjgl.opengl.GL11.GL_QUADS;
import static org.lwjgl.opengl.GL11.glBegin;
import static org.lwjgl.opengl.GL11.glClear;
import static org.lwjgl.opengl.GL11.glColor3f;
import static org.lwjgl.opengl.GL11.glEnd;
import static org.lwjgl.opengl.GL11.glMatrixMode;
import static org.lwjgl.opengl.GL11.glOrtho;
import static org.lwjgl.opengl.GL11.glVertex2f;

import java.io.IOException;
import java.nio.IntBuffer;
import java.util.List;

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
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.DisplayMode;
import org.lwjgl.opengl.GL11;
import org.newdawn.slick.opengl.Texture;
import org.newdawn.slick.opengl.TextureImpl;
import org.newdawn.slick.opengl.TextureLoader;
import org.newdawn.slick.util.ResourceLoader;

import common.Buffer;

public class MRU {
	final int nExecutors = 100;
	IntBuffer map, x, y, dir, mem, ptr;
	
	CLPlatform platform;
	List<CLDevice> devices;
	CLDevice device;
	CLContext context;
	CLCommandQueue queue;
	
	CLMem mapMem, xMem, yMem, dirMem, memMem, ptrMem;
	
	PointerBuffer kernel1DGlobalWorkSize;
	CLProgram program;
	CLKernel kernel;
	
	Texture texture;

	final int imageSizeX = 50, imageSizeY = 50;
	final int blokDimX = 16, blokDimY = 16;
	
	int time;

	void start() throws LWJGLException {
		initCL();
		initGL();
		loadTex();
		doTexCoords(blokDimX, blokDimY, blokDimX * 4, blokDimY * 4);
		loop();
		freeGL();
		freeCL();
	}
	
	void loadTex() {
		try {
			texture = TextureLoader.getTexture("PNG", ResourceLoader.getResourceAsStream("mru_texture.png"), GL11.GL_NEAREST);
		} catch(IOException e) {
			e.printStackTrace();
		}
	}
	
	void initGL() {
		try {
			Display.setDisplayMode(new DisplayMode(imageSizeX * blokDimX, imageSizeY * blokDimY));
			Display.setTitle("OpenCL Massive RU");
			Display.create();
		} catch(LWJGLException e) {
			e.printStackTrace();
			Display.destroy();
			System.exit(1);
		}

		GL11.glEnable(GL11.GL_TEXTURE_2D);
		//GL11.glEnable(GL11.GL_BLEND);
    //GL11.glBlendFunc(GL11.GL_SRC_ALPHA, GL11.GL_ONE_MINUS_SRC_ALPHA);
		
		glMatrixMode(GL_PROJECTION);
		glOrtho(0, imageSizeX * 10, imageSizeY * 10, 0, 1, -1);
	}
	
	void loop() {
		glMatrixMode(GL_MODELVIEW);

		while(!Display.isCloseRequested()) {
			stepCL();

			glClear(GL_COLOR_BUFFER_BIT);

			drawMap(blokDimX, blokDimY);
			drawAllParticles(blokDimX, blokDimY);
			
			Display.update();
			Display.sync(10);
		}
	}

	void drawAllParticles(int blokDimX, int blokDimY) {
		float tx, ty;
		
		TextureImpl.unbind();
		GL11.glBindTexture(GL11.GL_TEXTURE_2D, 0);
		glColor3f(0f, 0f, 0f);
		
		glBegin(GL_QUADS);

		for(int i = 0; i < nExecutors; i++) {
			tx = x.get(i) * blokDimX;
			ty = y.get(i) * blokDimX;

			glVertex2f(tx           , ty           );
			glVertex2f(tx + blokDimX, ty           );
			glVertex2f(tx + blokDimX, ty + blokDimY);
			glVertex2f(tx           , ty + blokDimY);
		}

		glEnd();
	}
	
	float[][] colo = {
			{0   , 0.07f, 0.1f},
			{0.4f,  0   , 0   }, {0.4f, 0.4f, 0   }, {0   , 0.4f, 0   }, {0   , 0.2f, 0.4f},
			{0.8f,  0.4f, 0.4f}, {0.8f, 0.8f, 0.4f}, {0.4f, 0.8f, 0.4f}, {0.4f, 0.6f, 0.8f},
			{0.9f,  0.6f, 0   }, {0.6f, 0.9f, 0   }, {0.9f, 0.2f, 0.6f}, {0.6f, 0.2f, 0.9f}};
	
	float[][] texCoordX, texCoordY;
	
	void doTexCoords(float blokDimX, float blokDimY, float texDimX, float texDimY) {
		texCoordX = new float[16][4];
		texCoordY = new float[16][4];
		
		final float rapX = blokDimX / texDimX;
		final float rapY = blokDimY / texDimY;
		
		final float m = 1f / 64f;
		
		for(int i = 0; i < 4; i++)
			for(int j = 0; j < 4; j++) {
				int k = j * 4 + i;
				
				texCoordX[k][0] =     i * rapX;
				texCoordX[k][1] = (i+1) * rapX - m;
				texCoordX[k][2] = (i+1) * rapX - m;
				texCoordX[k][3] =     i * rapX;
				
				texCoordY[k][0] =     j * rapY;
				texCoordY[k][1] =     j * rapY;
				texCoordY[k][2] = (j+1) * rapY - m;
				texCoordY[k][3] = (j+1) * rapY - m;
			}
	}
	
	void drawMap(int blokDimX, int blokDimY) {
		float tx, ty;

		glColor3f(1, 1, 1);
		texture.bind();
		
		glBegin(GL_QUADS);

		for(int i = 0; i < imageSizeX; i++)
			for(int j = 0; j < imageSizeY; j++) {
				int type = map.get(i * imageSizeX + j);
				
				tx = i * blokDimX;
				ty = j * blokDimY;

				GL11.glTexCoord2f(texCoordX[type][0], texCoordY[type][0]);
				//GL11.glTexCoord2f(0, 0);
				glVertex2f(tx           , ty           );
				GL11.glTexCoord2f(texCoordX[type][1], texCoordY[type][1]); 
				glVertex2f(tx + blokDimX-1, ty           );
				//GL11.glTexCoord2f(0, 1);
				GL11.glTexCoord2f(texCoordX[type][2], texCoordY[type][2]); 
				glVertex2f(tx + blokDimX-1, ty + blokDimY-1);
				//GL11.glTexCoord2f(1, 1);
				GL11.glTexCoord2f(texCoordX[type][3], texCoordY[type][3]); 
				glVertex2f(tx           , ty + blokDimY-1);
				//GL11.glTexCoord2f(1, 0);
			}

		glEnd();
	}
	
	void freeGL() {
		Display.destroy();
	}
	
	void initCL() throws LWJGLException {
		final String source = KernelBuilder.create(imageSizeX, imageSizeY, nExecutors);

		// buffers
		x = Buffer.toIntBuffer(Buffer.randomIntArray(nExecutors, 0, imageSizeX - 1));
		y = Buffer.toIntBuffer(Buffer.randomIntArray(nExecutors, 0, imageSizeY - 1));
		dir = Buffer.toIntBuffer(Buffer.zeroIntArray(nExecutors));
		mem = Buffer.toIntBuffer(Buffer.zeroIntArray(nExecutors));
		ptr = Buffer.toIntBuffer(Buffer.rangeIntArray(nExecutors, 0, 1));

		// CL creation
		CL.create();
		platform = CLPlatform.getPlatforms().get(0);
		devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
		context = CLContext.create(platform, devices, null, null, null);
		queue = clCreateCommandQueue(context, devices.get(0), CL_QUEUE_PROFILING_ENABLE, null);

		// memory management
		xMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, x, null);
		clEnqueueWriteBuffer(queue, xMem, 1, 0, x, null, null);
		yMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, y, null);
		clEnqueueWriteBuffer(queue, yMem, 1, 0, y, null, null);
		dirMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, dir, null);
		clEnqueueWriteBuffer(queue, dirMem, 1, 0, dir, null, null);
		memMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem, null);
		clEnqueueWriteBuffer(queue, memMem, 1, 0, mem, null, null);
		ptrMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ptr, null);
		clEnqueueWriteBuffer(queue, ptrMem, 1, 0, ptr, null, null);
		
		// map creation
		map = BufferUtils.createIntBuffer(imageSizeX * imageSizeY);
		for(int i = 0; i < imageSizeX * imageSizeY; i++) {
			if(Math.random() < 0.7) 
				map.put(0);
			else
				map.put((int)(Math.random() * 13));
		}
		map.rewind();

		mapMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, map, null);
		clEnqueueWriteBuffer(queue, mapMem, 1, 0, map, null, null);
		
		clFinish(queue);

		// kernel creation
		program = clCreateProgramWithSource(context, source, null);
		Util.checkCLError(clBuildProgram(program, devices.get(0), "", null));
		kernel = clCreateKernel(program, "adv", null);

		kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
		kernel1DGlobalWorkSize.put(0, nExecutors);
		
		kernel.setArg(0, mapMem);
		kernel.setArg(1, xMem);
		kernel.setArg(2, yMem);
		kernel.setArg(3, dirMem);
		kernel.setArg(4, memMem);
		kernel.setArg(5, ptrMem);
	}
	
	void stepCL() {
		clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);
		clEnqueueReadBuffer(queue, xMem, 1, 0, x, null, null);
		clEnqueueReadBuffer(queue, yMem, 1, 0, y, null, null);
		
		time++;
		if(time >= 10) {
			time = 0;
			clEnqueueReadBuffer(queue, memMem, 1, 0, mem, null, null);
			clEnqueueReadBuffer(queue, ptrMem, 1, 0, ptr, null, null);
			System.out.print("Mem: "); Buffer.print(mem);
			System.out.print("Ptr: "); Buffer.print(ptr);
		}
		
		clFinish(queue);
	}

	void freeCL() {
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		
		clReleaseMemObject(mapMem);
		clReleaseMemObject(xMem);
		clReleaseMemObject(yMem);
		clReleaseMemObject(dirMem);
		clReleaseMemObject(memMem);
		
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		CL.destroy();
	}
	
	public static void main(String args[])  throws LWJGLException {
		MRU instance = new MRU();
		instance.start();
		System.exit(0);
	}
}