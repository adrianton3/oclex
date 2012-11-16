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

public class KernelBuilder {
 static String create(RanGen ranGen, String function, float[][] dom, int nTries) {
 	String kernelSource = 
 			"float f(float x, float y) {" +
 		 " return " + function + "; " +
 		 "}" +
 	  "" +
 	  "int ran(int state) { " +
 	  " return (state * " + ranGen.a + " + " + ranGen.c + ") % " + ranGen.m + "; " +
 	  "} " +
 	  "\n" +
 	  "float toSub(int x) {" +
 	  " return ((float)x)/(float)" + ranGen.m + "; " +
 	  "}" +
 	  "\n" +
 	  "float toInterval(float x, float s, float e) {" +
 	  " return x*(e-s) + s;" +
 	  "}" +
 	  "\n" +
 	  "kernel void climb(global float *x, global float *y, global float *ans) { " +
 	  " unsigned int xid = get_global_id(0); " +
 	  " int rs = ran(xid*" + ranGen.aInit + " + " + ranGen.cInit + "); " +
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
 	  " while(tries < " + nTries + ") {" +
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
 	return kernelSource;
 }
}
