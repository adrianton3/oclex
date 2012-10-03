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

package pso.OF;


public class Rastrigin implements ObjectiveFunction {
 public String getCLStr() { 
 	final String tmpstr = "0";
  return "20 + (x-"+tmpstr+")*(x-"+tmpstr+") + (y-"+tmpstr+")*(y-"+tmpstr+") + cos(x*6.2831)*10 + cos(y*6.2831)*10";
 }
	
	public float f(float x, float y) {
		return (float)(20 + x*x + y*y + Math.cos(6.2831*x)*10 + Math.cos(6.2831*y)*10);
	}
}
