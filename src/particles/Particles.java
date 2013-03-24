/*
 * Copyright 2012 Adrian Toncean
 * 
 * This file is part of OCLEx.
 *
 * OCLEx is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * OCLEx is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with OCLEx. If not, see <http://www.gnu.org/licenses/>.
 */

package particles;

import java.nio.FloatBuffer;
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

import common.Buffer;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opengl.GL11.*;

public class Particles {
	final int nParticles = 100;
	FloatBuffer x;
	FloatBuffer y;
	FloatBuffer vx;
	FloatBuffer vy;

	CLPlatform platform;
	List<CLDevice> devices;
	CLContext context;
	CLCommandQueue queue;

	CLMem xMem, yMem, vxMem, vyMem;

	PointerBuffer kernel1DGlobalWorkSize;
	CLProgram program;
	CLKernel kernel;

	void start() throws LWJGLException {
		initCL();
		initGL();
		loop();
		freeGL();
		freeCL();
	}

	void initGL() {
		try {
			Display.setDisplayMode(new DisplayMode(800, 600));
			Display.setTitle("OpenCL Particle Demo");
			Display.create();
		} catch(LWJGLException e) {
			e.printStackTrace();
			Display.destroy();
			System.exit(1);
		}

		glMatrixMode(GL_PROJECTION);
		glOrtho(0, 800, 600, 0, 1, -1);
	}

	void loop() {
		glMatrixMode(GL_MODELVIEW);

		while(!Display.isCloseRequested()) {
			stepCL();

			glClear(GL_COLOR_BUFFER_BIT);

			drawAllParticles();

			Display.update();
			Display.sync(30);
		}
	}

	void drawAllParticles() {
		float tx, ty;
		int i;

		glBegin(GL_QUADS);
		glColor3f(1.0f, 0.0f, 0.0f);

		for(i = 0; i < nParticles; i++) {
			tx = x.get(i);
			ty = y.get(i);

			glVertex2f(tx, ty);
			glVertex2f(tx + 3, ty);
			glVertex2f(tx + 3, ty + 3);
			glVertex2f(tx, ty + 3);
		}

		glEnd();
	}

	void freeGL() {
		Display.destroy();
	}

	void initCL() throws LWJGLException {
		final float minDist = 50;
		final String source = KernelBuilder.create(nParticles, minDist);

		// buffers
		x = Buffer.toFloatBuffer(Buffer.randomFloatArray(nParticles, 0, 800));
		y = Buffer.toFloatBuffer(Buffer.randomFloatArray(nParticles, 0, 600));
		vx = Buffer.toFloatBuffer(Buffer.randomFloatArray(nParticles, -10, 10));
		vy = Buffer.toFloatBuffer(Buffer.randomFloatArray(nParticles, -10, 10));

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
		vxMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vx, null);
		clEnqueueWriteBuffer(queue, vxMem, 1, 0, vx, null, null);
		vyMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vy, null); // CL_MEM_WRITE_ONLY
		clEnqueueWriteBuffer(queue, vyMem, 1, 0, vy, null, null);
		clFinish(queue);

		// kernel creation
		program = clCreateProgramWithSource(context, source, null);
		Util.checkCLError(clBuildProgram(program, devices.get(0), "", null));
		kernel = clCreateKernel(program, "adv", null);

		kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
		kernel1DGlobalWorkSize.put(0, nParticles);
		kernel.setArg(0, xMem);
		kernel.setArg(1, yMem);
		kernel.setArg(2, vxMem);
		kernel.setArg(3, vyMem);
	}

	void stepCL() {
		clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);
		clEnqueueReadBuffer(queue, xMem, 1, 0, x, null, null);
		clEnqueueReadBuffer(queue, yMem, 1, 0, y, null, null);
		clFinish(queue);
	}

	void freeCL() {
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseMemObject(xMem);
		clReleaseMemObject(yMem);
		clReleaseMemObject(vxMem);
		clReleaseMemObject(vyMem);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		CL.destroy();
	}

	public static void main(String[] args) throws LWJGLException {
		Particles instance = new Particles();
		instance.start();
		System.exit(0);
	}
}