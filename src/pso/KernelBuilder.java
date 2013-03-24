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

import pso.of.ObjectiveFunction;

public class KernelBuilder {
 private static String createNetwork(PSOParam psoParam) {
  String ret = "";
		for (int i = 1; i < psoParam.connectivity; i++)
  		ret += "\n" +
 					" rs = ran(rs); " +
 					" tmp1 = ranToInt(rs,0,99); " +
 					" if(fit[tmp1] < fit[besti]) besti = tmp1; ";
  return ret;
 }
 
 static String create(RanGen ranGen, PSOParam psoParam, ObjectiveFunction of) {
 	String kernelSource = 
 			"float f(float x, float y) {" +
 				 " return " + of.getCLStr() + "; " +
 				 "}" +
 				 "\n" +
 				 "int ran(int state) { " +
 				 " return (state * " + ranGen.a + " + " + ranGen.c + ") % " + ranGen.m + "; " +
 				 "} " +
 				 "\n" +
 				 "float toSub(int x) {" +
 				 " return ((float)x)/(float)" + ranGen.m + "; " +
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
 	    " int rs = ran(xid*" + ranGen.aInit + " + " + ranGen.cInit + "); " +
 	    " int tmp1, tmp2; " +
 	    " int besti; " +
 	    "\n" +
 	    " fit[xid] = f(x[xid],y[xid]); " +
 	    "\n" +
 					" rs = ran(rs); " +
 					" besti = ranToInt(rs,0,99); " +
 					"\n" +
 					" rs = ran(rs); " +
 					" tmp1 = ranToInt(rs,0,99); " +
 					" if(fit[tmp1] < fit[besti]) besti = tmp1; " +
 					
 					createNetwork(psoParam) +
 					/*
 					"\n" +
 					" rs = ran(rs); " +
 					" tmp1 = ranToInt(rs,0,99); " +
 					" if(fit[tmp1] < fit[besti]) besti = tmp1; " +
 					"\n" +
 					" rs = ran(rs); " +
 					" tmp1 = ranToInt(rs,0,99); " +
 					" if(fit[tmp1] < fit[besti]) besti = tmp1; " +
 					*/
 	    "\n" +
 	    "  vx[xid] = vx[xid]*" + psoParam.atenuator + " + (x[besti] - x[xid])*" + psoParam.social + "; " +
 	    "  vy[xid] = vy[xid]*" + psoParam.atenuator + " + (y[besti] - y[xid])*" + psoParam.social + "; " +
 	    "  x[xid] += vx[xid]; " +
 	    "  y[xid] += vy[xid]; " +
 	    "}"
 	    ;
 	
 	return kernelSource;
 }
}