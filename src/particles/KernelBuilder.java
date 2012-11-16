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

public class KernelBuilder {
 static String create(int nParticles, float mindist) {
 	final String kernelSource =
    "kernel void " +
    "adv(global float *x, " +
    "    global float *y, " +
    "    global float *vx," +
    "    global float *vy) { " +
    " unsigned int xid = get_global_id(0); " +
    " unsigned int xid2 = (xid + 1) % " + nParticles + "; " +
    "\n" +
    " int i; " +
    " float dx, dy; " +
    " float dist, rap; " +
    " vx[xid] *= 0.7; " +
    " vy[xid] *= 0.7; " +
    "\n" +
    " for(i=0;i<" + nParticles + ";i++) { " +
    "  dx = x[i] - x[xid]; " +
    "  dy = y[i] - y[xid]; " +
    "  dist = sqrt(dx*dx + dy*dy); " +
    "\n" +
    "  if(dist < " + mindist + ") { " +
    "   rap = (" + mindist + "-dist)/" + mindist + "; " +
    "   vx[xid] -= dx * rap * 0.02; " +
    "   vy[xid] -= dy * rap * 0.02; " + 
    "  } " +
    "\n" +
    " } " +
    " vx[xid] += (x[xid2] - x[xid])*0.05; " +
    " vy[xid] += (y[xid2] - y[xid])*0.05; " +
    " x[xid] += vx[xid]; " +
    " y[xid] += vy[xid]; " +
    " if(x[xid] < 0) x[xid] = 0; else if(x[xid] > 800) x[xid] = 800; " +
    " if(y[xid] < 0) y[xid] = 0; else if(y[xid] > 600) y[xid] = 600; " +
    "}";
 	return kernelSource;
 }
}
