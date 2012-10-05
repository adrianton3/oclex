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

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opengl.GL11.*;

public class Particles {
	final int np = 100;
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
   Display.setDisplayMode(new DisplayMode(800,600));
   Display.setTitle("OpenCL Particle Demo");
   Display.create();
  } catch (LWJGLException e) {
   e.printStackTrace();
   Display.destroy();
   System.exit(1);
  }
		
		glMatrixMode(GL_PROJECTION);
  glOrtho(0, 800, 600, 0, 1, -1);
	}
	
	void loop() {
  glMatrixMode(GL_MODELVIEW);

  while (!Display.isCloseRequested()) {
  	stepCL();
  	
  	glClear(GL_COLOR_BUFFER_BIT);
  	
  	drawFunction();
  	drawAllParticles();
  	
  	Display.update();
  	Display.sync(30);
  }
	}
	
	void drawFunction() {
		
	}
	
	void drawAllParticles() {
		glBegin(GL_QUADS);
		glColor3f(1.0f, 0.0f, 0.0f);
		
		float tx, ty;
		int i;
		for(i=0;i<np;i++) {
			tx = x.get(i);
			ty = y.get(i);
			
   glVertex2f(tx,ty);
   glVertex2f(tx+3,ty);
   glVertex2f(tx+3,ty+3);
   glVertex2f(tx,ty+3);
		}
  
		glEnd();
	}
	
	void freeGL() {
		Display.destroy();
	}
	
	void initCL() throws LWJGLException {
		//kernel source code
		/*
		final String source =
    "kernel void " +
    "adv(global float *x, " +
    "    global float *y, " +
    "    global float *vx," +
    "    global float *vy) { " +
    "  unsigned int xid = get_global_id(0); " +
    "  x[xid] += vx[xid]; " +
    "  y[xid] += vy[xid]; " +
    "  if(x[xid] < 0 || x[xid] > 800) vx[xid] *= -1; " +
    "  if(y[xid] < 0 || y[xid] > 600) vy[xid] *= -1; " +
    "}"
    ;
		*/
		/*
		final String source =
    "kernel void " +
    "adv(global float *x, " +
    "    global float *y, " +
    "    global float *vx," +
    "    global float *vy) { " +
    "  unsigned int xid = get_global_id(0); " +
    "  unsigned int xid2 = (xid+1) % 100; " +
    "  unsigned int xid3 = (xid+2) % 100; " +
    "  unsigned int xid4 = (xid*3+24) % 100; " +
    "  vx[xid] = vx[xid]*0.75 + (x[xid2] - x[xid])*0.04 + (400 - x[xid])*0.001 + 4*sign(x[xid] - x[xid3])/((x[xid] - x[xid3])*(x[xid] - x[xid3])+0.4); " +
    "  vy[xid] = vy[xid]*0.75 + (y[xid2] - y[xid])*0.04 + (300 - y[xid])*0.001 + 4*sign(y[xid] - y[xid3])/((x[xid] - x[xid3])*(y[xid] - y[xid3])+0.4); " +
    "  x[xid] += vx[xid]; " +
    "  y[xid] += vy[xid]; " +
    "  if(x[xid] < 0) x[xid] = 0; else if(x[xid] > 800) x[xid] = 800; " +
    "  if(y[xid] < 0) y[xid] = 0; else if(y[xid] > 600) y[xid] = 600; " +
    "  if(x[xid] < 0 || x[xid] > 800) vx[xid] *= -1; " +
    "  if(y[xid] < 0 || y[xid] > 600) vy[xid] *= -1; " +
    "}"
    ; */
  final float mindist = 100;
  
		final String source =
    "kernel void " +
    "adv(global float *x, " +
    "    global float *y, " +
    "    global float *vx," +
    "    global float *vy) { " +
    "  unsigned int xid = get_global_id(0); " +
    "\n" +
    " int i; " +
    " float dx, dy; " +
    " float dist, rap; " +
    " vx[xid] *= 0.98; " +
    " vy[xid] *= 0.98; " +
    "\n" +
    " for(i=0;i<"+np+";i++) { " +
    "  dx = x[i] - x[xid]; " +
    "  dy = y[i] - y[xid]; " +
    "  dist = sqrt(dx*dx + dy*dy); " +
    "\n" +
    "  if(dist < "+mindist+") { " +
    "   rap = ("+mindist+"-dist)/"+mindist+"; " +
    "   vx[xid] -= dx * rap * 0.2; " +
    "   vy[xid] -= dy * rap * 0.2; " + 
    "  } " +
    "\n" +
    " } " +
    " x[xid] += vx[xid]; " +
    " y[xid] += vy[xid]; " +
    " if(x[xid] < 0) x[xid] = 0; else if(x[xid] > 800) x[xid] = 800; " +
    " if(y[xid] < 0) y[xid] = 0; else if(y[xid] > 600) y[xid] = 600; " +
    //" if(x[xid] < 0 || x[xid] > 800) vx[xid] *= -1; " +
    //" if(y[xid] < 0 || y[xid] > 600) vy[xid] *= -1; " +
    "}"
    ;
		
		//buffers
		x = toFloatBuffer(randomFloatArray(np,0,800));
  y = toFloatBuffer(randomFloatArray(np,0,600));
  vx = toFloatBuffer(randomFloatArray(np,-10,10));
  vy = toFloatBuffer(randomFloatArray(np,-10,10));
  
  //cl creation
  CL.create();
  platform = CLPlatform.getPlatforms().get(0);
  devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
  context = CLContext.create(platform, devices, null, null, null);
  queue = clCreateCommandQueue(context, devices.get(0), CL_QUEUE_PROFILING_ENABLE, null);
  
  xMem = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, x, null);
  clEnqueueWriteBuffer(queue, xMem, 1, 0, x, null, null);
  yMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, y, null);
  clEnqueueWriteBuffer(queue, yMem, 1, 0, y, null, null);
  vxMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vx, null);
  clEnqueueWriteBuffer(queue, vxMem, 1, 0, vx, null, null);
  vyMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vy, null); //CL_MEM_WRITE_ONLY
  clEnqueueWriteBuffer(queue, vyMem, 1, 0, vy, null, null);
  clFinish(queue);
  
  //kernel creation
  program = clCreateProgramWithSource(context,source,null);
  Util.checkCLError(clBuildProgram(program,devices.get(0),"",null));
  kernel = clCreateKernel(program,"adv",null);
  
  kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
  kernel1DGlobalWorkSize.put(0,np);
  kernel.setArg(0,xMem);
  kernel.setArg(1,yMem);
  kernel.setArg(2,vxMem);
  kernel.setArg(3,vyMem);
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
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  CL.destroy();
	}
	
	static float[] randomFloatArray(int n, float s, float e) {
  float[] ret = new float[n];
  
  int i;
  for(i=0;i<ret.length;i++)
  	ret[i] = (float)(Math.random()*(e-s)+s);
  
  return ret;
 }
	
	static FloatBuffer toFloatBuffer(float[] floats) {
  FloatBuffer buf = BufferUtils.createFloatBuffer(floats.length).put(floats);
  buf.rewind();
  return buf;
 }
	
	public static void main(String[] args) throws LWJGLException {
  Particles instance = new Particles();
  instance.start();
  System.exit(0);
	}
}