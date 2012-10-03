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

package pso;

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

import pso.OF.ObjectiveFunction;
import pso.OF.Rastrigin;

import static org.lwjgl.opencl.CL10.*;
import static org.lwjgl.opengl.GL11.*;

public class PSO {
	final int pso_np = 100;
	FloatBuffer x;
	FloatBuffer y;
	FloatBuffer vx;
	FloatBuffer vy;
	FloatBuffer fit;
	
	CLPlatform platform;
 List<CLDevice> devices;
 CLContext context;
 CLCommandQueue queue;
	
	CLMem xMem, yMem, vxMem, vyMem, fitMem;
	
	PointerBuffer kernel1DGlobalWorkSize;
	CLProgram program;
	CLKernel kernel;
	
	final ObjectiveFunction of = new Rastrigin();
	final float[][] dom = {{-5.12f, 5.12f},{-5.12f, 5.12f}};
	final int rezx = 200;
	final int rezy = 200;
	
	void start() throws LWJGLException {
		initCL();
		initGL();
		initSurface();
  loop();
  Surface.release();
  freeGL();
  freeCL();
	}
	
	void initGL() {
		try {
   Display.setDisplayMode(new DisplayMode(600,600));
   Display.setTitle("OpenCL Particle Swarm Optimization Demo");
   Display.create();
  } catch (LWJGLException e) {
   e.printStackTrace();
   Display.destroy();
   System.exit(1);
  }
		
		glMatrixMode(GL_PROJECTION);
  //glOrtho(0, 800, 600, 0, 1, -1);
		glOrtho(dom[0][0], dom[0][1], dom[1][1], dom[1][0], 1, -1);
	}
	
	void initSurface() {
		Val val = compute(of,dom,rezx,rezy);
		Surface.assemble(val.v,val.minc,val.maxc,rezx,rezy);
	}
	
	Val compute(ObjectiveFunction of, float[][] dom, int rezx, int rezy) {
  float[][] ret = new float[rezx][rezy];
  float pasx, pasy;
  float px, py;
  pasx = (dom[0][1] - dom[0][0]) / rezx;
  pasy = (dom[1][1] - dom[1][0]) / rezy;
  float maxc = -10000, minc = 10000;
  int i, j;

  px = dom[0][0];
  for(i=0;i<rezx;i++)
  {
   py = dom[0][0];
   for(j=0;j<rezy;j++)
   {
    ret[i][j] = of.f(px,py);
    
    if(ret[i][j] < minc) minc = ret[i][j];
    else if(ret[i][j] > maxc) maxc = ret[i][j];
    
    py += pasy;
   }
   px += pasx;
  }
  
  return new Val(ret,minc,maxc);
 }
	
	void loop() {
  glMatrixMode(GL_MODELVIEW);

  while (!Display.isCloseRequested()) {
  	stepCL();
  	
  	glClear(GL_COLOR_BUFFER_BIT);
  	
  	glPushMatrix();
  	glScalef((dom[0][1]-dom[0][0])/(float)rezx,(dom[1][1]-dom[1][0])/(float)rezy,1.0f);
  	Surface.call();
  	glPopMatrix();
  	
  	drawAllParticles();
  	
  	Display.update();
  	Display.sync(30);
  }
	}
	
	void drawAllParticles() {
		glBegin(GL_QUADS);

		glColor3f(0.0f, 0.0f, 0.0f);
		
		float tx, ty;
		int i;
		for(i=0;i<pso_np;i++) {
			tx = x.get(i);
			ty = y.get(i);
			
   glVertex2f(tx-0.05f,ty-0.05f);
   glVertex2f(tx+0.05f,ty-0.05f);
   glVertex2f(tx+0.05f,ty+0.05f);
   glVertex2f(tx-0.05f,ty+0.05f);
		}
		
		glEnd();
	}
	
	void freeGL() {
		Display.destroy();
	}
	
	void initCL() throws LWJGLException {
		final int rangen_p1 = 3418;
  final int rangen_p2 = 2349;
  final int rangen_init_p1 = 3418;
  final int rangen_init_p2 = 2349;
  final int rangen_m = 9871;
  
  final String function = of.getCLStr();
  
  final int pso_niter = 1;
  final float pso_atenuator = 0.8f;
  final float pso_social = 0.7f;
  final float pso_personal = 0.4f;
  
		final String source =
				"float f(float x, float y) {" +
			 " return " + function + "; " +
			 "}" +
			 "\n" +
			 "int ran(int state) { " +
			 " return (state * "+rangen_p1+" + "+rangen_p2+") % "+rangen_m+"; " +
			 "} " +
			 "\n" +
			 "float toSub(int x) {" +
			 " return ((float)x)/(float)"+rangen_m+"; " +
			 "}" +
			 "\n" +
			 "float toInterval(float x, float s, float e) { " +
			 " return x*(e-s) + s; " +
			 "}" +
			 "\n" +
			 "int ranToInt(int state, int st, int en) { " +
			 " return state % (en-st) + st;" +
			 "}" +
			 "\n" +
    "kernel void " +
    "adv(global float *x, " +
    "    global float *y, " +
    "    global float *vx," +
    "    global float *vy," +
    "    global float *fit) { " +
    " unsigned int xid = get_global_id(0); " +
    " int rs = ran(xid*" + rangen_init_p1 + " + " + rangen_init_p2 + "); " +
    " float x_bestp, y_bestp; " +
    " float fit_bestp = 10000; " +
    " int tmp1, tmp2; " +
    " int besti; " +
    "\n" +
    " int iter = 0; " +
  //  " while(iter < " + pso_niter + ") {" +
    "  iter++; " +
    "\n" +
    "  fit[xid] = f(x[xid],y[xid]); " +
    "  if(fit[xid] < fit_bestp) { " +
    "   fit_bestp = fit[xid]; " +
    "   x_bestp = x[xid]; " +
    "   y_bestp = y[xid]; " +
    "  } " +
    "\n" +
    "  rs = ran(rs); " +                  //this part can be expanded
    "  tmp1 = ranToInt(rs,0,99); " +
    "  rs = ran(rs); " +
    "  tmp2 = ranToInt(rs,0,99); " +      //...
    "  if(fit[tmp1] < fit[tmp2]) { " +
    "   besti = tmp1; " +
    "  }" +
    "  else {" +
    "   besti = tmp2; " +
    "  } " +
    "\n" +
    "  vx[xid] = vx[xid]*0.6 + (x[besti] - x[xid])*0.3 + (x_bestp - x[xid])*0.2; " +
    "  vy[xid] = vy[xid]*0.6 + (y[besti] - y[xid])*0.3 + (y_bestp - y[xid])*0.2; " +
    "  x[xid] += vx[xid]; " +
    "  y[xid] += vy[xid]; " +
  //  " }" +
    "}"
    ;
		
		//buffers
		x = toFloatBuffer(randomFloatArray(pso_np,dom[0][0],dom[0][1]));
  y = toFloatBuffer(randomFloatArray(pso_np,dom[1][0],dom[1][1]));
  vx = toFloatBuffer(randomFloatArray(pso_np,dom[0][0]/10f,dom[0][1]/10f));
  vy = toFloatBuffer(randomFloatArray(pso_np,dom[1][0]/10f,dom[1][1]/10f));
  fit = toFloatBuffer(zeroFloatArray(pso_np));
  
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
  vyMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vy, null);
  clEnqueueWriteBuffer(queue, vyMem, 1, 0, vy, null, null);
  fitMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, fit, null);
  clEnqueueWriteBuffer(queue, fitMem, 1, 0, fit, null, null);
  clFinish(queue);
  
  //kernel creation
  program = clCreateProgramWithSource(context,source,null);
  Util.checkCLError(clBuildProgram(program,devices.get(0),"",null));
  kernel = clCreateKernel(program,"adv",null);
  
  kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
  kernel1DGlobalWorkSize.put(0,pso_np);
  kernel.setArg(0,xMem);
  kernel.setArg(1,yMem);
  kernel.setArg(2,vxMem);
  kernel.setArg(3,vyMem);
  kernel.setArg(4,fitMem);
	}
	
	void stepCL() {
  clEnqueueNDRangeKernel(queue, kernel, 1, null, kernel1DGlobalWorkSize, null, null, null);
  clEnqueueReadBuffer(queue, xMem, 1, 0, x, null, null);
  clEnqueueReadBuffer(queue, yMem, 1, 0, y, null, null);
  clEnqueueReadBuffer(queue, fitMem, 1, 0, fit, null, null);
  clFinish(queue);
	}
	
	void freeCL() {
		clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  CL.destroy();
	}
	
	static float[] zeroFloatArray(int n) {
  float[] ret = new float[n];
  
  int i;
  for(i=0;i<ret.length;i++)
  	ret[i] = 0f;
  
  return ret;
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
  PSO instance = new PSO();
  instance.start();
  System.exit(0);
	}
}