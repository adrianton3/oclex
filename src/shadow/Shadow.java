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

package shadow;

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

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opengl.GL11.*;

public class Shadow {
	final int np = 400;
	final int nsegs = 10;
	FloatBuffer x;
	FloatBuffer y;
	FloatBuffer ax, ay;
	FloatBuffer bx, by;
	FloatBuffer vx, vy;

	CLPlatform platform;
	List<CLDevice> devices;
	CLContext context;
	CLCommandQueue queue;

	CLMem xMem, yMem, axMem, ayMem, bxMem, byMem, vxMem, vyMem;

	PointerBuffer kernel1DGlobalWorkSize1, kernel1DGlobalWorkSize2;
	CLProgram program1, program2;
	CLKernel kernel1, kernel2;

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
			Display.setTitle("OpenCL Shadows Demo");
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
		float tx, ty, tax, tay, tbx, tby;
		int i;

		glBegin(GL_LINES);
		glColor3f(1.0f, 0.0f, 0.0f);
		for(i = 0; i < np; i++) {
			tx = x.get(i);
			ty = y.get(i);

			glVertex2f(400, 300);
			glVertex2f(tx, ty);
		}
		glEnd();

		glBegin(GL_LINES);
		glColor3f(1.0f, 1.0f, 1.0f);
		for(i = 0; i < nsegs; i++) {
			tax = ax.get(i);
			tay = ay.get(i);
			tbx = bx.get(i);
			tby = by.get(i);

			glVertex2f(tax, tay);
			glVertex2f(tbx, tby);
		}
		glEnd();
	}

	void freeGL() {
		Display.destroy();
	}

	TwoFloatAr radialBeam(float cx, float cy, float r, int nr) {
		float[] x, y;
		x = new float[nr];
		y = new float[nr];
		float k, ak;
		k = 0;
		ak = (float) (Math.PI * 2f / nr);
		int i;
		for(i = 0; i < nr; i++) {
			x[i] = (float) (Math.cos(k) * r) + cx;
			y[i] = (float) (Math.sin(k) * r) + cy;
			k += ak;
		}
		return new TwoFloatAr(x, y);
	}

	FourFloatAr columns(int nx, int ny) {
		final float sx = 0, sy = 0, ex = 800, ey = 600;
		float[] ax, ay, bx, by;
		ax = new float[nx * ny];
		ay = new float[nx * ny];
		bx = new float[nx * ny];
		by = new float[nx * ny];

		int i;
		for(i = 0; i < nx * ny; i++) {
			ax[i] = (float) (Math.random() * (ex - sx) + sx);
			ay[i] = (float) (Math.random() * (ey - sy) + sy);
			if(Math.random() < 0.5) {
				bx[i] = ax[i];
				by[i] = ay[i] + 50;
			} else {
				bx[i] = ax[i] + 50;
				by[i] = ay[i];
			}
		}

		return new FourFloatAr(ax, ay, bx, by);
	}

	void initCL() throws LWJGLException {
		final float cx = 400;
		final float cy = 300;

		final String source1 = "float dp(float ax, float ay, float bx, float by) {" + " return ax*bx + ay*by; " +
				"}" + "\n" + "float4 intersect(float p0_x, float p0_y, float p1_x, float p1_y, " +
				"                 float p2_x, float p2_y, float p3_x, float p3_y) { " +
				" float s1_x, s1_y, s2_x, s2_y; " + " s1_x = p1_x - p0_x; s1_y = p1_y - p0_y; " +
				" s2_x = p3_x - p2_x; s2_y = p3_y - p2_y; " + "\n" + " float s, t; " +
				" s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y); " +
				" t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y); " + "\n" +
				" return (float4)(p0_x + t*s1_x,p0_y + t*s1_y,s,t); " + "}" + "\n" + "kernel void " +
				"adv(global float *x, " + "    global float *y, " + "    global float *ax, " +
				"    global float *ay, " + "    global float *bx, " + "    global float *by) { " +
				" unsigned int xid = get_global_id(0); " + "\n" + " x[xid] = 400 + 500*cos(xid*(2*3.1415/" + np +
				")); " + " y[xid] = 300 + 500*sin(xid*(2*3.1415/" + np + ")); " + " int i; " + " for(i=0;i<" + nsegs +
				";i++) { " + "  float4 tmp = intersect(" + cx + "," + cy +
				",x[xid],y[xid],ax[i],ay[i],bx[i],by[i]); " +
				"  if((0 <= tmp.z) && (tmp.z <= 1) && (0 <= tmp.w) && (tmp.w <= 1)) { " + "   x[xid] = tmp.x; " +
				"   y[xid] = tmp.y; " + "  } " + " } " + "}";

		final String source2 = "kernel void " + "segadv(global float *ax, " + "    global float *ay, "
				+ "    global float *bx, " + "    global float *by, " + "    global float *vx, "
				+ "    global float *vy) { " + " unsigned int xid = get_global_id(0); " + "\n"
				+ " if(ax[xid] > 800 || ax[xid] < 0 || bx[xid] > 800 || bx[xid] < 0) vx[xid] = -vx[xid]; "
				+ " if(ay[xid] > 600 || ay[xid] < 0 || by[xid] > 600 || by[xid] < 0) vy[xid] = -vy[xid]; "
				+ " ax[xid] += vx[xid]; " + " ay[xid] += vy[xid]; " + " bx[xid] += vx[xid]; "
				+ " by[xid] += vy[xid]; " + "}";

		TwoFloatAr rb = radialBeam(400, 300, 500, np);
		FourFloatAr column = columns(1, nsegs);
		// buffers
		x = toFloatBuffer(rb.x);
		y = toFloatBuffer(rb.y);
		ax = toFloatBuffer(column.ax);
		ay = toFloatBuffer(column.ay);
		bx = toFloatBuffer(column.bx);
		by = toFloatBuffer(column.by);
		vx = toFloatBuffer(randomFloatArray(nsegs, -2, 2));
		vy = toFloatBuffer(randomFloatArray(nsegs, -2, 2));

		// cl creation
		CL.create();
		platform = CLPlatform.getPlatforms().get(0);
		devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
		context = CLContext.create(platform, devices, null, null, null);
		queue = clCreateCommandQueue(context, devices.get(0), CL_QUEUE_PROFILING_ENABLE, null);

		xMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, x, null);
		clEnqueueWriteBuffer(queue, xMem, 1, 0, x, null, null);
		yMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, y, null);
		clEnqueueWriteBuffer(queue, yMem, 1, 0, y, null, null);

		axMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ax, null);
		clEnqueueWriteBuffer(queue, axMem, 1, 0, ax, null, null);
		ayMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ay, null);
		clEnqueueWriteBuffer(queue, ayMem, 1, 0, ay, null, null);

		bxMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bx, null);
		clEnqueueWriteBuffer(queue, bxMem, 1, 0, bx, null, null);
		byMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, by, null);
		clEnqueueWriteBuffer(queue, byMem, 1, 0, by, null, null);

		vxMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vx, null);
		clEnqueueWriteBuffer(queue, vxMem, 1, 0, vx, null, null);
		vyMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vy, null);
		clEnqueueWriteBuffer(queue, vyMem, 1, 0, vy, null, null);

		clFinish(queue);

		// kernel creation
		program1 = clCreateProgramWithSource(context, source1, null);
		Util.checkCLError(clBuildProgram(program1, devices.get(0), "", null));
		kernel1 = clCreateKernel(program1, "adv", null);

		kernel1DGlobalWorkSize1 = BufferUtils.createPointerBuffer(1);
		kernel1DGlobalWorkSize1.put(0, np);
		kernel1.setArg(0, xMem);
		kernel1.setArg(1, yMem);
		kernel1.setArg(2, axMem);
		kernel1.setArg(3, ayMem);
		kernel1.setArg(4, bxMem);
		kernel1.setArg(5, byMem);

		// kernel 2
		program2 = clCreateProgramWithSource(context, source2, null);
		Util.checkCLError(clBuildProgram(program2, devices.get(0), "", null));
		kernel2 = clCreateKernel(program2, "segadv", null);

		kernel1DGlobalWorkSize2 = BufferUtils.createPointerBuffer(1);
		kernel1DGlobalWorkSize2.put(0, nsegs);
		kernel2.setArg(0, axMem);
		kernel2.setArg(1, ayMem);
		kernel2.setArg(2, bxMem);
		kernel2.setArg(3, byMem);
		kernel2.setArg(4, vxMem);
		kernel2.setArg(5, vyMem);
	}

	void stepCL() {
		clEnqueueNDRangeKernel(queue, kernel1, 1, null, kernel1DGlobalWorkSize1, null, null, null);
		clEnqueueReadBuffer(queue, xMem, 1, 0, x, null, null);
		clEnqueueReadBuffer(queue, yMem, 1, 0, y, null, null);

		clEnqueueNDRangeKernel(queue, kernel2, 1, null, kernel1DGlobalWorkSize2, null, null, null);
		clEnqueueReadBuffer(queue, axMem, 1, 0, ax, null, null);
		clEnqueueReadBuffer(queue, ayMem, 1, 0, ay, null, null);
		clEnqueueReadBuffer(queue, bxMem, 1, 0, bx, null, null);
		clEnqueueReadBuffer(queue, byMem, 1, 0, by, null, null);
		clFinish(queue);
	}

	void freeCL() {
		clReleaseKernel(kernel1);
		clReleaseProgram(program1);

		clReleaseKernel(kernel2);
		clReleaseProgram(program2);
		
		clReleaseMemObject(xMem);
		clReleaseMemObject(yMem);
		clReleaseMemObject(axMem);
		clReleaseMemObject(ayMem);
		clReleaseMemObject(bxMem);
		clReleaseMemObject(byMem);

		clReleaseCommandQueue(queue);
		clReleaseContext(context);
		CL.destroy();
	}

	static float[] randomFloatArray(int n, float s, float e) {
		float[] ret = new float[n];

		int i;
		for(i = 0; i < ret.length; i++)
			ret[i] = (float) (Math.random() * (e - s) + s);

		return ret;
	}

	static FloatBuffer toFloatBuffer(float[] floats) {
		FloatBuffer buf = BufferUtils.createFloatBuffer(floats.length).put(floats);
		buf.rewind();
		return buf;
	}

	public static void main(String[] args) throws LWJGLException {
		Shadow instance = new Shadow();
		instance.start();
		System.exit(0);
	}
}