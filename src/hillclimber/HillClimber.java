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

package hillclimber;

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
import static org.lwjgl.opencl.CL10.clFinish;
import static org.lwjgl.opencl.CL10.clReleaseCommandQueue;
import static org.lwjgl.opencl.CL10.clReleaseContext;
import static org.lwjgl.opencl.CL10.clReleaseKernel;
import static org.lwjgl.opencl.CL10.clReleaseMemObject;
import static org.lwjgl.opencl.CL10.clReleaseProgram;

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

import common.Buffer;
import common.ValIndex;

public class HillClimber {
	final int nClimbers = 100;
	FloatBuffer x, y, ans;

	CLPlatform platform;
	List<CLDevice> devices;
	CLDevice device;
	CLContext context;
	CLCommandQueue queue;

	CLMem xMem, yMem, ansMem;

	PointerBuffer kernel1DGlobalWorkSize;
	CLProgram program;
	CLKernel kernel;

	void start() throws Exception {
		initCL();
		runCL();
		processResults();
		freeCL();
	}

	String packKernel() {
		final RanGen ranGen = new RanGen();
		final float[][] dom = { { -5.12f, 5.12f }, { -5.12f, 5.12f } };
		final int nTries = 100;
		final String function = "x*x + y*y";

		final String source = KernelBuilder.create(ranGen, function, dom, nTries);

		return source;
	}

	void initCL() throws LWJGLException {
		final String source = packKernel();

		// buffers
		x = BufferUtils.createFloatBuffer(nClimbers);
		y = BufferUtils.createFloatBuffer(nClimbers);
		ans = BufferUtils.createFloatBuffer(nClimbers);

		// cl creation
		CL.create();
		platform = CLPlatform.getPlatforms().get(0);
		devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
		device = devices.get(0);
		context = CLContext.create(platform, devices, null, null, null);
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, null);

		xMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, x, null);
		yMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, y, null);
		ansMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ans, null);
		clFinish(queue);

		// kernel creation
		program = clCreateProgramWithSource(context, source, null);
		Util.checkCLError(clBuildProgram(program, device, "", null));
		kernel = clCreateKernel(program, "climb", null);

		kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
		kernel1DGlobalWorkSize.put(0, nClimbers);
		kernel.setArg(0, xMem);
		kernel.setArg(1, yMem);
		kernel.setArg(2, ansMem);
	}

	void runCL() {
		clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);
		clEnqueueReadBuffer(queue, xMem, 1, 0, x, null, null);
		clEnqueueReadBuffer(queue, yMem, 1, 0, y, null, null);
		clEnqueueReadBuffer(queue, ansMem, 1, 0, ans, null, null);
		clFinish(queue);
	}

	void freeCL() {
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseMemObject(xMem);
		clReleaseMemObject(yMem);
		clReleaseMemObject(ansMem);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		CL.destroy();
	}

	void processResults() throws Exception {
		System.out.print("x: ");
		Buffer.print(x);
		System.out.print("y: ");
		Buffer.print(y);
		System.out.print("f(x,y): ");
		Buffer.print(ans);

		ValIndex pv = Buffer.min(ans);
		System.out.println("Min: ((" + x.get(pv.index) + "," + y.get(pv.index) + "), " + pv.val + ")");
	}

	public static void main(String[] args) throws Exception {
		HillClimber instance = new HillClimber();
		instance.start();
		System.exit(0);
	}
}
