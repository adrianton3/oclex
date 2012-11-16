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

package common;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import org.lwjgl.BufferUtils;

public class Buffer {
	public static FloatBuffer toFloatBuffer(float[] floats) {
  FloatBuffer buf = BufferUtils.createFloatBuffer(floats.length).put(floats);
  buf.rewind();
  return buf;
 }
   
	public static IntBuffer toIntBuffer(int[] ints) {
  IntBuffer buf = BufferUtils.createIntBuffer(ints.length).put(ints);
  buf.rewind();
  return buf;
 }
	
	public static ValIndex min(FloatBuffer buffer) throws Exception {
		if (buffer.capacity() <= 0)	throw new Exception("buffer.capacity() <= 0");
		if (buffer.capacity() == 1)	return new ValIndex(buffer.get(),1);

		float min = buffer.get(0);
		int mini = 0;
		float tmp;

		for (int i = 0; i < buffer.capacity(); i++) {
			tmp = buffer.get(i);
			if (tmp < min) {
				min = tmp;
				mini = i;
			}
		}

		return new ValIndex(min, mini);
	}

	public static float[] zeroFloatArray(int n) {
  float[] ret = new float[n];
  
  int i;
  for(i=0;i<ret.length;i++)
  	ret[i] = 0f;
  
  return ret;
 }
	
	public static float[] randomFloatArray(int n, float s, float e) {
  float[] ret = new float[n];
  
  int i;
  for(i=0;i<ret.length;i++)
  	ret[i] = (float)(Math.random()*(e-s)+s);
  
  return ret;
 }
	
 public static void print(FloatBuffer buffer) {
  for (int i = 0; i < buffer.capacity(); i++) {
   System.out.print(buffer.get(i)+" ");
  }
  System.out.println();
 }
   
 public static void print(IntBuffer buffer) {
  for (int i = 0; i < buffer.capacity(); i++) {
   System.out.print(buffer.get(i)+" ");
  }
  System.out.println();
 }
}
