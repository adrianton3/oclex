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

import org.lwjgl.opencl.Util;
import org.lwjgl.opencl.CLMem;
import org.lwjgl.opencl.CLCommandQueue;
import org.lwjgl.BufferUtils;
import org.lwjgl.LWJGLException;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CLProgram;
import org.lwjgl.opencl.CLKernel;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.List;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CLContext;
import org.lwjgl.opencl.CLDevice;
import org.lwjgl.opencl.CLPlatform;

import static org.lwjgl.opencl.CL10.*;

public class HillClimber {
	final int np = 100;
	FloatBuffer x, y, ans;
	
	CLPlatform platform;
 List<CLDevice> devices;
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
		final int rangen_p1 = 3418;
  final int rangen_p2 = 2349;
  final int rangen_init_p1 = 3418;
  final int rangen_init_p2 = 2349;
  final int m = 9871;
  
  final float[][] dom = {{-5.12f, 5.12f},{-5.12f, 5.12f}};
  final int tries = 100;
  final String function = "x*x + y*y";
 	
 	final String source =
   "float f(float x, float y) {" +
   " return " + function + "; " +
   "}" +
   "" +
   "int ran(int state) { " +
   " return (state * "+rangen_p1+" + "+rangen_p2+") % "+m+"; " +
   "} " +
   "\n" +
   "float toSub(int x) {" +
   " return ((float)x)/(float)"+m+"; " +
   "}" +
   "\n" +
   "float toInterval(float x, float s, float e) {" +
   " return x*(e-s) + s;" +
   "}" +
   "\n" +
   "kernel void climb(global float *x, global float *y, global float *ans) { " +
   " unsigned int xid = get_global_id(0); " +
   " int rs = ran(xid*" + rangen_init_p1 + " + " + rangen_init_p2 + "); " +
   "\n" +
   " const float sx = " + dom[0][0] + ", ex = " + dom[0][1] + "; " +
   " const float sy = " + dom[1][0] + ", ey = " + dom[1][1] + "; " +
   "\n" +
   " float lx, ly;" +
   "\n" +
   " rs = ran(rs); " +
   " lx = toInterval(toSub(rs),sx,ex); " +
   " rs = ran(rs); " +
   " ly = toInterval(toSub(rs),sy,ey); " +
   "\n" +
   " float px, py; " +
   " rs = ran(rs); " +
   " px = toInterval(toSub(rs),-0.1,0.1); " +
   " rs = ran(rs); " +
   " py = toInterval(toSub(rs),-0.1,0.1); " +
   "\n" +
   " float nx, ny, min = 1000, tmin;" +
   " int tries = 0; " +
   " while(tries < " + tries + ") {" +
   "  tries++;" +
   "\n" +
   "  nx = lx + px; " +
   "  ny = ly + py; " +
   "  tmin = f(nx,ny); " +
   "  if(tmin < min) {" +
   "   min = tmin; " +
   "   lx = nx; " +
   "   ly = ny; " +
   "  } else {" +
   "   rs = ran(rs); " +
   "   px = toInterval(toSub(rs),-0.2,0.2); " +
   "   rs = ran(rs); " +
   "   py = toInterval(toSub(rs),-0.2,0.2); " +
   "  }" +
   " } " +
   "\n" +
   " x[xid] = lx; " +
   " y[xid] = ly; " +
   " ans[xid] = f(lx,ly); " +
   "}";
 	
 	return source;
	}

 void initCL() throws LWJGLException {
  final String source = packKernel();
   		
   //buffers
   x = BufferUtils.createFloatBuffer(np);
   y = BufferUtils.createFloatBuffer(np);
   ans = BufferUtils.createFloatBuffer(np);
     
   //cl creation
   CL.create();
   platform = CLPlatform.getPlatforms().get(0);
   devices = platform.getDevices(CL_DEVICE_TYPE_GPU);
   context = CLContext.create(platform, devices, null, null, null);
   queue = clCreateCommandQueue(context, devices.get(0), CL_QUEUE_PROFILING_ENABLE, null);
    
   xMem = clCreateBuffer(context,CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, x, null);
   yMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, y, null);
   ansMem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, ans, null);
   clFinish(queue);
    
   //kernel creation
   program = clCreateProgramWithSource(context,source,null);
   Util.checkCLError(clBuildProgram(program,devices.get(0),"",null));
   kernel = clCreateKernel(program,"climb",null);
    
   kernel1DGlobalWorkSize = BufferUtils.createPointerBuffer(1);
   kernel1DGlobalWorkSize.put(0,np);
   kernel.setArg(0,xMem);
   kernel.setArg(1,yMem);
   kernel.setArg(2,ansMem);
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
   clReleaseCommandQueue(queue);
   clReleaseContext(context);
   CL.destroy();
  }
  
  void processResults() throws Exception {
  	System.out.print("x: ");	print(x);
  	System.out.print("y: ");	print(y);
  	System.out.print("f(x,y): ");	print(ans);
  	
  	PosVal pv = min(ans);
  	System.out.println("Min: ((" + x.get(pv.pos) + "," + y.get(pv.pos) + "), " + pv.val +")");
  }
  
  static PosVal min(FloatBuffer buffer) throws Exception {
   if(buffer.capacity() <= 0) throw new Exception("buffer.capacity() <= 0");
   if(buffer.capacity() == 1) return new PosVal(1,buffer.get());
   
   float min = buffer.get(0);
   int mini = 0;
   float tmp;
   
  	for (int i = 0; i < buffer.capacity(); i++) {
    tmp = buffer.get(i);
    if(tmp < min) {
    	min = tmp;
    	mini = i;
    }
   }
  	
   return new PosVal(mini,min);
  }

  static FloatBuffer toFloatBuffer(float[] floats) {
   FloatBuffer buf = BufferUtils.createFloatBuffer(floats.length).put(floats);
   buf.rewind();
   return buf;
  }
    
  static IntBuffer toIntBuffer(int[] ints) {
   IntBuffer buf = BufferUtils.createIntBuffer(ints.length).put(ints);
   buf.rewind();
   return buf;
  }

  static void print(FloatBuffer buffer) {
   for (int i = 0; i < buffer.capacity(); i++) {
    System.out.print(buffer.get(i)+" ");
   }
   System.out.println("");
  }
    
  static void print(IntBuffer buffer) {
	  for (int i = 0; i < buffer.capacity(); i++) {
	   System.out.print(buffer.get(i)+" ");
	  }
	  System.out.println("");
  }

	public static void main(String[] args) throws Exception {
	 HillClimber instance = new HillClimber();
	 instance.start();
	 System.exit(0);
	}
}

class PosVal {
	final int pos;
	final float val;
	
	PosVal(int pos,float val) {
		this.pos = pos;
		this.val = val;
	}
}
